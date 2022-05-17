import os
import json
import random
from optparse import OptionParser
from collections import defaultdict

import numpy as np

# Script to take the output of the label aggregation model and create multiple folds of data for training and evaluation


def main():
    usage = "%prog texts.json est_item_probs.json"
    parser = OptionParser(usage=usage)
    parser.add_option('--partitions', type=int, default=5,
                      help='Number of partitions to create: default=%default')
    parser.add_option('--test', type=int, default=300,
                      help='Number of test examples per partition: default=%default')
    parser.add_option('--dev', type=int, default=400,
                      help='Number of dev examples per partition: default=%default')
    parser.add_option('--seed', type=int, default=42,
                      help='Random seed: default=%default')
    parser.add_option('--basedir', type=str, default='splits',
                      help='Base output directory: default=%default')
    parser.add_option('--extra-data-file', type=str, default=None,
                      help='.jsonlist file with extra labeled examples (e.g., from MFC): default=%default')
    parser.add_option('--extra-weight', type=float, default=0.5,
                      help='optionally downweight extra examples: default=%default')

    (options, args) = parser.parse_args()

    data_file = args[0]
    item_probs_file = args[1]     # estimated label probabilities from label aggregation

    np.random.seed(options.seed)
    random.seed(options.seed)

    partitions = options.partitions
    n_test = options.test
    n_dev = options.dev
    basedir = options.basedir
    extra_data_file = options.extra_data_file
    extra_weight = options.extra_weight

    extra_train_lines = []
    if extra_data_file is not None:
        with open(extra_data_file) as f:
            lines = f.readlines()
        extra_train_lines = [json.loads(line) for line in lines]
    for line in extra_train_lines:
        line['weight'] = line['weight'] * extra_weight

    # Load the annotations (one annotation per pine)
    print("Loading data")
    with open(data_file) as f:
        data = json.load(f)
    print(len(data))

    # load the item probs
    with open(item_probs_file) as f:
        item_probs = json.load(f)
    print(len(item_probs))

    assert len(data) > len(item_probs)

    item_ids = sorted(item_probs)

    for f in range(partitions):
        # evenly partition the test data

        test_ids = list(np.random.choice(item_ids, size=n_test, replace=False))
        remaining_ids = sorted(list(set(item_ids) - set(test_ids)))
        dev_ids = list(np.random.choice(remaining_ids, size=n_dev, replace=False))
        train_ids = list(set(item_ids) - set(test_ids) - set(dev_ids))
        print(len(item_ids), len(train_ids), len(dev_ids), len(test_ids))

        for weighting in ['basic', 'label-weight']:

            if weighting == 'label-weight':
                use_label_weight = True
                duplicate = False
                outdir = os.path.join(basedir, 'tone', 'label-weight', 'folds',  str(f))
            else:
                use_label_weight = False
                duplicate = False
                outdir = os.path.join(basedir, 'tone', 'basic', 'folds',  str(f))

            if not os.path.exists(outdir):
                os.makedirs(outdir)

            write_to_file(outdir, 'train', train_ids, data, item_probs, use_label_weight=use_label_weight, train=True, duplicate=duplicate, extra_train_lines=extra_train_lines)

            write_to_file(outdir, 'dev', dev_ids, data, item_probs, train=False)

            write_to_file(outdir, 'test', test_ids, data, item_probs, train=False)

            write_to_file(outdir, 'nontest', train_ids + dev_ids, data, item_probs, use_label_weight=use_label_weight, train=True, duplicate=duplicate, extra_train_lines=extra_train_lines)

    # Finally output everything as training data
    for weighting in ['basic', 'label-weight']:

        if weighting == 'label-weight':
            use_label_weight = True
            duplicate = False
            outdir = os.path.join(basedir, 'tone', 'label-weight', 'all')
        else:
            use_label_weight = False
            duplicate = False
            outdir = os.path.join(basedir, 'tone', 'basic', 'all')

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        write_to_file(outdir, 'all', item_ids, data, item_probs, use_label_weight=use_label_weight, train=True, duplicate=duplicate, extra_train_lines=extra_train_lines)


def write_to_file(outdir, output_prefix, item_ids, data, item_probs, use_label_weight=False, train=False,  duplicate=False, extra_train_lines=None):

    outlines = []
    outlines_json = []
    label_counts = defaultdict(int)

    classes = ['anti', 'neutral', 'pro']

    for item_id in item_ids:
        text = data[item_id]['text']
        text = text.strip()
        class_probs = item_probs[item_id]

        if train:
            if duplicate:
                # add all labels with corresponding weights
                for c_i, label in enumerate(classes):
                    weight = class_probs[c_i]
                    outlines.append(text + '\t' + label + '\t' + str(weight) + '\n')
                    outlines_json.append({'id': item_id, 'text': text, 'tokens': data[item_id]['tokens'], 'label': label, 'weight': weight})
            else:
                label = classes[int(np.argmax(class_probs))]
                label_prob = np.max(class_probs)
                weight = 1.
                if use_label_weight:
                    weight *= label_prob
                outlines.append(text + '\t' + label + '\t' + str(weight) + '\n')
                outlines_json.append({'id': item_id, 'text': text, 'tokens': data[item_id]['tokens'], 'label': label, 'weight': weight})
                label_counts[label] += 1

        else:
            label = classes[int(np.argmax(class_probs))]
            weight = 1.0
            outlines.append(text + '\t' + label + '\t' + str(weight) + '\n')
            outlines_json.append({'id': item_id, 'text': text, 'tokens': data[item_id]['tokens'], 'label': label, 'weight': weight})
            label_counts[label] += 1

    if extra_train_lines is not None:
        outlines_json.extend(extra_train_lines)

    if train:
        random.shuffle(outlines_json)

    print(label_counts)

    outfile = os.path.join(outdir, output_prefix + '.jsonlist')
    with open(outfile, 'w') as fo:
        for line in outlines_json:
            fo.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
