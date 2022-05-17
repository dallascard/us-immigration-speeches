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
""" processors and helpers """

import os
import json
import logging

from transformers.file_utils import is_tf_available
from transformers.data.processors.utils import InputExample, InputFeatures
from hf.utils import InputFeaturesWeighted
import numpy as np

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)
# hdlr = logging.FileHandler('./tmp/logger.log')
# formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
# hdlr.setFormatter(formatter)
# logger.addHandler(hdlr)


class InputExampleWeighted(InputExample):
    """
    Inherit from InputExample but add a weight field
    """

    def __init__(self, guid, text_a, text_b=None, label=None, weight=1.0):
        super().__init__(guid, text_a, text_b, label)
        self.weight = weight


def convert_examples_to_features(
        examples,
        tokenizer,
        label_list,
        output_mode='classification',
        max_length=512):

    if max_length is None:
        max_length = tokenizer.max_len

    label_map = {label: i for i, label in enumerate(label_list)}

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        example = examples[i]
        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        feature = InputFeaturesWeighted(**inputs, label=label, weight=example.weight)
        #feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    example = examples[0]
    logger.info("*** Example 0 ***")
    logger.info("guid: %s" % (examples[0].guid))
    logger.info('example text_a: %s' % (example.text_a))
    logger.info('example text_b: %s' % (example.text_b))
    logger.info("input_ids: %s" % " ".join([str(x) for x in features[0].input_ids]))
    logger.info("attention_mask: %s" % " ".join([str(x) for x in features[0].attention_mask]))
    #logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
    logger.info("label: %s (id = %d)" % (example.label, features[0].label))

    return features


class GenericSingleProcessorWeighted(object):
    """Processor for classification."""

    def __init__(self, label_list, text_field='text', text_field_b=None, label_field='label', weight_field=None):
        self._label_list = label_list
        self._text_field = text_field
        self._text_field_b = text_field_b
        self._label_field = label_field
        self._weight_field = weight_field

        print("Constructing processor:")
        print("Label list:", self._label_list)
        print("Label field:", self._label_field)
        print("Text fields:", self._text_field, self._text_field_b)
        print("Weight field:", self._weight_field)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExampleWeighted(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            tensor_dict["target"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
            tensor_dict['weight'].numpy()
        )

    def get_train_examples(self, data_file):
        """See base class."""
        return self._create_examples(self._read_jsonl(data_file), "train")

    def get_dev_examples(self, data_file):
        """See base class."""
        return self._create_examples(self._read_jsonl(data_file), "dev")

    def get_test_examples(self, data_file, default_label=None):
        """See base class."""
        return self._create_examples(self._read_jsonl(data_file), "test", default_label=default_label)

    def get_examples(self, data_file, partition, default_label=None):
        """See base class."""
        return self._create_examples(self._read_jsonl(data_file), partition, default_label)

    def get_labels(self):
        """See base class."""
        return self._label_list

    def _create_examples(self, lines, set_type, default_label=None):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[self._text_field]
            text_b = None
            if self._text_field_b is not None:
                text_b = line[self._text_field_b]
            if default_label is None:
                try:
                    label = line[self._label_field]
                except IndexError as e:
                    print(i, line)
                    raise e
            else:
                label = default_label

            # use 1.0 as a default weight if not given
            if self._weight_field is None:
                weight = 1.0
            else:
                weight = float(line[self._weight_field])
            examples.append(InputExampleWeighted(guid=guid, text_a=text_a, text_b=text_b, label=label, weight=weight))
        return examples

    def _read_jsonl(cls, input_file, encoding='utf-8'):
        """Reads a tab separated value file."""
        with open(input_file, 'r', encoding=encoding) as f:
            return [json.loads(line) for line in f]

