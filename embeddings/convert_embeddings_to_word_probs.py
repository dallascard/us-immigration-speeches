import os
import re
import json
from glob import glob
from collections import Counter, defaultdict
from optparse import OptionParser

import torch
import numpy as np
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer, BertForMaskedLM, RobertaForMaskedLM
#from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.modeling_bert import BertOnlyMLMHead
from scipy.special import softmax

from analysis.metaphor_terms import get_metaphor_terms


# Export token occurrences (masked) to vector occurrences using Roberta


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--infile', type=str, default='data/speeches/Congress/contextual-embeddings/bert-base-uncased/immigrant_vectors_masked.npz',
                      help='Model type: default=%default')
    parser.add_option('--random-file', type=str, default='data/speeches/Congress/basic_counts/random_nouns.txt',
                      help='List of random nouns (from analysis.get_random_nouns.py): default=%default')
    parser.add_option('--model-type', type=str, default='bert',
                      help='Model type: default=%default')
    parser.add_option('--tokenizer', type=str, default=None,
                      help='Tokenizer name or path (defaults to model): default=%default')
    parser.add_option('--model', type=str, default='bert-base-uncased',
                      help='Model name or path: default=%default')
    parser.add_option('--device', type=int, default=0,
                      help='GPU to use: default=%default')
    parser.add_option('--batch-size', type=int, default=256,
                      help='Batch size: default=%default')
    parser.add_option('--outdir', type=str, default='data/speeches/Congress/contextual-embeddings/bert-base-uncased/',
                      help='Model name or path: default=%default')

    (options, args) = parser.parse_args()

    infile = options.infile
    random_file = options.random_file    
    tokenizer = options.tokenizer
    model_type = options.model_type
    model_name_or_path = options.model
    device = options.device
    batch_size = options.batch_size
    outdir = options.outdir

    metaphor_terms = get_metaphor_terms()
    combined = []
    for category, terms in metaphor_terms.items():
        combined.extend(terms)
    metaphor_terms['combined'] = combined

    with open(random_file) as f:
        random_nouns = f.readlines()
    metaphor_terms['random'] = [t.strip() for t in random_nouns]
    metaphor_terms['random_xhuman'] = [t for t in metaphor_terms['random'] if t != 'humans']

    print("Loading model")
    if model_type == 'bert':
        model_class = BertModel
        tokenizer_class = BertTokenizer
        lm_model_class = BertForMaskedLM
        lm_head_class = BertOnlyMLMHead
    elif model_type == 'roberta':
        model_class = RobertaModel
        tokenizer_class = RobertaTokenizer
        lm_model_class = RobertaForMaskedLM
        raise NotImplementedError("Need to get the lm_head_class for Roberta")
    else:
        raise ValueError("Model type not recognized")

    # Load pretrained model/tokenizer
    if tokenizer is None:
        tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    else:
        tokenizer = tokenizer_class.from_pretrained(tokenizer)

    #model = model_class.from_pretrained(model_name_or_path)
    lm = lm_model_class.from_pretrained(model_name_or_path)

    # move the model to the GPU
    torch.cuda.set_device(device)
    device = torch.device("cuda", device)
    lm.to(device)

    vocab = tokenizer.vocab

    for category, target_terms in metaphor_terms.items():
        print(category)
        target_indices = []
        found_terms = []
        for term in target_terms:
            if term in vocab:
                target_indices.append(vocab[term])
                found_terms.append(term)
        print("Using found target terms:")
        print(' '.join(sorted(found_terms)))

        # get the MLM head
        mlm = None
        for m in lm.modules():
            if type(m) == lm_head_class:
                mlm = m

        embeddings = np.load(infile)['vectors']
        n_words, dim = embeddings.shape

        n_batches = n_words // batch_size + 1
        print(n_batches)

        all_target_prob_sums = []
        all_target_log_prob_sums = []
        for b in tqdm(range(n_batches)):
            batch = embeddings[b * batch_size: (b + 1) * batch_size, :]
            batch = torch.tensor(batch).to(device)
            preds = mlm(batch)
            probs = torch.softmax(preds, dim=1)
            probs_np = probs.detach().cpu().numpy().copy()
            target_probs_sum = np.sum(probs_np[:, target_indices], axis=1)
            target_log_probs_sum = np.log(np.sum(probs_np[:, target_indices], axis=1))
            all_target_prob_sums.extend(list(target_probs_sum))
            all_target_log_prob_sums.extend(list(target_log_probs_sum))

        outfile = os.path.join(outdir, 'metaphor_probs_' + category + '.npz')
        np.savez(outfile, probs=np.array(all_target_prob_sums))

        outfile = os.path.join(outdir, 'metaphor_log_probs_' + category + '.npz')
        np.savez(outfile, log_probs=np.array(all_target_log_prob_sums))


if __name__ == '__main__':
    main()
