# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import os
import glob
import json
import random
import logging
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    FlaubertConfig,
    FlaubertForSequenceClassification,
    FlaubertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)

from hf.metrics import compute_metrics
from hf.processors import convert_examples_to_features, GenericSingleProcessorWeighted

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    "flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
}



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, classes):
    """ Create file to log loss, etc. """
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    log_file = open(os.path.join(args.output_dir, "logs.tsv"),'w+')
    log_file.write("Key\tValue\tStep\n")

    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
        )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    try:
        if os.path.exists(args.model_name_or_path):
            # set global_step to gobal_step of last saved checkpoint from model path
            #logger.info('Model name or path:',args.model_name_or_path,args.model_name_or_path.split("-"))
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
    except Exception as e:
        pass

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            labels = batch[2].to(args.device)
            instance_weights = batch[3].to(args.device)
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            #if args.model_type != "distilbert":
            #    inputs["token_type_ids"] = (
            #        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
            #    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            logits = outputs[0]

            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.view(-1, args.num_labels), labels.view(-1))
            loss = (loss * instance_weights).mean()

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    #if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    #if True:
                    logs = {}
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, tokenizer, classes)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        #tb_writer.add_scalar(key, value, global_step)
                        log_file.write("{}\t{}\t{}\n".format(key, value, global_step))
                        #logger.info("Value of %s",key)
                        #logger.info(value)
                    #print(json.dumps({**logs, **{"step": global_step}}))

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.local_rank in [-1, 0]:
            # Save model checkpoint after each epoch
            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            print('output dir:',output_dir)
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)

            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s", output_dir)


        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    log_file.close()
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, processor, classes, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    #eval_task_names = ("mnli", "mnli-mm") if args.name == "mnli" else (args.name,)
    metrics = args.metrics.split(',')
    print(metrics)
    eval_output_dir = args.output_dir
    task = args.name

    results = {}
    eval_dataset = load_and_cache_examples(args, task, processor, tokenizer, evaluate=True)

    #if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
    #    os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    instance_weights = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
            #if args.model_type != "distilbert":
            #    inputs["token_type_ids"] = (
            #        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
            #    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            #eval_loss += tmp_eval_loss.mean().item(
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            instance_weights = batch[3].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            instance_weights = np.r_[instance_weights, batch[3].detach().cpu().numpy()]

    #eval_loss = eval_loss / nb_eval_steps
    if args.output_mode == "classification":
        probs = np.vstack(preds)
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)

    print("Instance weight shape:", instance_weights.shape)
    print(preds.shape)
    print(out_label_ids.shape)
    # TODO: update to deal with more than 2 classes for f1
    result = compute_metrics(metrics, preds, out_label_ids, classes, instance_weights)
    results.update(result)

    preds_df = pd.DataFrame({'true': out_label_ids, 'predicted': preds})
    if args.output_mode == 'classification':
        n_instances, num_labels = probs.shape
        for label_num in range(num_labels):
            preds_df[label_num] = probs[:, label_num]
    #np.savez('preds.npz', preds=preds)
    eval_partition = args.eval_partition
    # preds_df = pd.DataFrame({'true':out_label_ids,
    # 'predicted':preds})
    # preds_df.to_csv(os.path.join(eval_output_dir, prefix, '{}.tsv'.format(args.pred_file_name)),sep='\t',index=False)
    #preds_df.to_csv(args.output_dir+'/{}.tsv'.format(args.pred_file_name),sep='\t',index=False)
    if eval_partition == 'test':
        file_prefix = os.path.splitext(args.test)[0]
    elif eval_partition == 'dev':
        file_prefix = os.path.splitext(args.dev)[0]
    else:
        file_prefix = eval_partition

    preds_df.to_csv(args.output_dir+'/{}_'.format(args.pred_file_name) + file_prefix + '.tsv', sep='\t', index=False)
    if prefix == '':
        output_eval_file = os.path.join(eval_output_dir, "eval_results_" + file_prefix + ".txt")
    else:
        output_eval_file = os.path.join(eval_output_dir, "eval_results_" + file_prefix + '_' + prefix + ".txt")

    #output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results_{}.txt".format(args.pred_file_name))
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def predict(args, model, tokenizer, processor, pred_target, prefix=""):
    eval_output_dir = args.output_dir
    task = args.name

    results = {}
    pred_dataset = load_and_cache_examples(args, task, processor, tokenizer, evaluate=False, pred_target=pred_target)

    #if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
    #    os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(pred_dataset)
    eval_dataloader = DataLoader(pred_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(pred_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            #if args.model_type != "distilbert":
            #    inputs["token_type_ids"] = (
            #        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
            #    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            logits = outputs[0]
            #eval_loss += tmp_eval_loss.mean().item(
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

    #eval_loss = eval_loss / nb_eval_steps
    if args.output_mode == "classification":
        probs = np.vstack(preds)
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)

    print(preds.shape)

    preds_df = pd.DataFrame()
    preds_df['predicted'] = preds
    if args.output_mode == 'classification':
        n_instances, num_labels = probs.shape
        for label_num in range(num_labels):
            preds_df[label_num] = probs[:, label_num]
    #np.savez('preds.npz', preds=preds)
    #eval_partition = args.eval_partition
    # preds_df = pd.DataFrame({'true':out_label_ids,
    # 'predicted':preds})
    preds_df.to_csv(os.path.join(eval_output_dir, prefix, '{}.tsv'.format(args.pred_file_name)), sep='\t', index=False)
    """
    #preds_df.to_csv(args.output_dir+'/{}.tsv'.format(args.pred_file_name),sep='\t',index=False)
    preds_df.to_csv(args.output_dir+'/{}_'.format(args.pred_file_name) + eval_partition + '.tsv', sep='\t', index=False)
    if prefix == '':
        output_eval_file = os.path.join(eval_output_dir, "eval_results_" + eval_partition + ".txt")
    else:
        output_eval_file = os.path.join(eval_output_dir, "eval_results_" + eval_partition + '_' + prefix + ".txt")

    #output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results_{}.txt".format(args.pred_file_name))
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    return results
    """

def load_and_cache_examples(args, task, processor, tokenizer, evaluate=False, pred_target=None):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file

    if args.eval_partition == 'test':
        file_prefix = args.test
    elif args.eval_partition == 'dev':
        file_prefix = args.dev
    else:
        file_prefix = args.eval_partition

    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "train" if not evaluate else file_prefix,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    print('cached_features_file:', cached_features_file)
    print(os.path.exists(cached_features_file))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        print("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]

        if evaluate:
            if args.eval_partition == 'test':
                if args.test_data_dir is None:
                    examples = (processor.get_test_examples(os.path.join(args.data_dir, args.test)))
                else:
                    examples = (processor.get_test_examples(os.path.join(args.test_data_dir, args.test)))
            elif args.eval_partition == 'dev':
                examples = (processor.get_dev_examples(os.path.join(args.data_dir, args.dev)))
            else:
                examples = (processor.get_examples(os.path.join(args.data_dir, args.eval_partition)))
                #raise RuntimeError("eval partition {:s} not recognized".format(args.eval_parition))
            print('num examples:', len(examples))
            print('example:')
            print(examples[0])
        elif pred_target is not None:
            print("Loading data for prediction from", pred_target)
            examples = (processor.get_examples(pred_target, 'pred', default_label=args.default_label))
            print('num examples:', len(examples))
            print('example:')
            print(examples[0])
        else:
            examples = processor.get_train_examples(os.path.join(args.data_dir, args.train))
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            output_mode=args.output_mode,
            max_length=args.max_seq_length,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    # Dropping this becuase not all use it; may need to deal with it for distilbert
    #all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if args.output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif args.output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    else:
        raise ValueError("Output mode {:s} not recognized".format(args.output_mode))
    all_weights = torch.tensor([f.weight for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels, all_weights)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: ",
    )
    parser.add_argument(
        "--name",
        default=None,
        type=str,
        required=True,
        help="The name of the setup",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="Directory with the data",
    )
    parser.add_argument(
        "--train",
        default='train.jsonlist',
        type=str,
        required=False,
        help="The name of the train file in data-dir",
    )
    parser.add_argument(
        "--dev",
        default='dev.jsonlist',
        type=str,
        required=False,
        help="The name of the dev file in data-dir",
    )
    parser.add_argument(
        "--test",
        default='test.jsonlist',
        type=str,
        required=False,
        help="The name of the test file in data-dir",
    )
    parser.add_argument(
        "--test_data_dir",
        default=None,
        type=str,
        required=False,
        help="Alternative directory with test data",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--text_field",
        default='text',
        type=str,
        required=False,
        help="Text field (a) in json objects",
    )
    parser.add_argument(
        "--text_field_b",
        default=None,
        type=str,
        required=False,
        help="Text field (b) in json objects",
    )
    parser.add_argument(
        "--label_field",
        default='label',
        type=str,
        required=False,
        help="Label field in json objects",
    )
    parser.add_argument(
        "--weight_field",
        default=None,
        type=str,
        required=False,
        help="Weight field in json objects (if any)",
    )
    parser.add_argument(
        "--metrics",
        default='simple_accuracy',
        type=str,
        required=False,
        help="Metrics (comma-separated) [accuracy,weighted_accuracy,f1,weighted_f1,micro_f1,macro_f1,per_class_f1,cfm",
    )

    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval.")
    #parser.add_argument("--eval_on_test", action="store_true", help="Whether to run eval on the dev set (False) or test set (True).")
    parser.add_argument(
        "--eval_partition",
        default='dev',
        type=str,
        help="Partition to evaluate on [dev|test|...].",
    )
    #parser.add_argument("--predict", action="store_true", help="Do prediction on a separate file.")
    parser.add_argument(
        "--predict",
        default=None,
        type=str,
        help="File name on which to do prediction",
    )
    parser.add_argument(
        "--default-label",
        default=None,
        type=str,
        help="Label to use as default is missing from eval_partition",
    )
    #parser.add_argument(
    #    "--add_placeholder_labels", action="store_true", help="Add placeholder labels (for unlabeled test data).",
    #)
    parser.add_argument("--pred_file_name", default='preds', help="Whether to run eval on the dev set (False) or test set (True).")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare data

    label_counter = Counter()

    with open(os.path.join(args.data_dir, args.train)) as f:
        for line in f:
            label_counter.update([json.loads(line)[args.label_field]])
    label_list = sorted(label_counter)
    print(label_list)
    args.num_labels = len(label_list)

    #if args.add_placeholder_labels:
    #    default_label = label_list[0]
    #    print("Using", default_label, "as placeholder label")
    #else:
    #    default_label = None

    if args.predict is not None and args.default_label is None:
        args.default_label = label_list[0]
        print("Using", args.default_label, "as default label")

    processor = GenericSingleProcessorWeighted(label_list=label_list,
                                               text_field=args.text_field,
                                               text_field_b=args.text_field_b,
                                               label_field=args.label_field,
                                               weight_field=args.weight_field)
    args.output_mode = 'classification'
    label_list = processor.get_labels()
    num_labels = len(label_list)
    print('Num labels:', num_labels)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    print("config", args.config_name if args.config_name else args.model_name_or_path)
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    print("tokenizer", args.tokenizer_name if args.tokenizer_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    print("model", args.model_name_or_path)
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
        #loss_fct=args.loss_fct,
    )
    #logger.info("Model loss function: %s", model.loss_fct)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.name, processor, tokenizer, evaluate=False, pred_target=None)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, label_list)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        print("Do evaluation:")
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            print('checkpoint:', checkpoint)
            model = model_class.from_pretrained(checkpoint, num_labels=num_labels)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, processor, label_list, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    elif args.predict is not None and args.local_rank in [-1, 0]:
        print("Doing prediction")
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        checkpoints = [args.model_name_or_path]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            print('checkpoint:', checkpoint)
            model = model_class.from_pretrained(checkpoint, num_labels=num_labels)
            model.to(args.device)
            predict(args, model, tokenizer, processor, args.predict, prefix=prefix)
            #result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            #results.update(result)

    # Plot loss curve
    """
    df = pd.read_csv(os.path.join(args.output_dir, 'logs.tsv'), sep='\t', header=0, index_col=False)
    loss_df = df[df.Key == 'loss']
    lr_df = df[df.Key == 'learning_rate']
    print(loss_df.head())
    print(lr_df.head())
    print(loss_df.shape)
    print(lr_df.shape)
    """

    #%matplotlib inline
    # fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    # loss_df.plot(x='Step',  y='Value', ax=ax1)
    # ax1.set_ylabel('Loss')
    # lr_df.plot(x='Step', y='Value', ax=ax2)
    # ax2.set_ylabel('Learning rate')
    # #plt.show()
    # plt.tight_layout()
    # fig.savefig(os.path.join(args.output_dir,'loss_lr_curves.png'))

    return results


if __name__ == "__main__":
    main()
