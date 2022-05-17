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
    parser.add_option('--partitions', type=int, default=10,
                      help='Number of partitions to create: default=%default')
    parser.add_option('--test', type=int, default=300,
                      help='Number of test examples per partition: default=%default')
    parser.add_option('--dev', type=int, default=400,
                      help='Number of dev examples per partition: default=%default')
    parser.add_option('--seed', type=int, default=42,
                      help='Random seed: default=%default')
    parser.add_option('--basedir', type=str, default='splits',
                      help='Base output directory: default=%default')

    (options, args) = parser.parse_args()

    data_file = args[0]
    item_probs_file = args[1]     # estimated label probabilities from label aggregation

    np.random.seed(options.seed)
    random.seed(options.seed)

    partitions = options.partitions
    n_test = options.test
    n_dev = options.dev
    basedir = options.basedir

    # Load the texts
    print("Loading data")
    with open(data_file) as f:
        data = json.load(f)
    print(len(data))

    # load the item probs
    with open(item_probs_file) as f:
        item_probs = json.load(f)
    print(len(item_probs))

    print(len(data))
    print(len(item_probs))

    item_ids = sorted(item_probs)

    #phase_two_ids = [item_id for item_id in item_ids if data[item_id]['phase'] == 2]
    phase_one_ids = [item_id for item_id in item_ids if data[item_id]['phase'] == 1]
    print(len(phase_one_ids))

    # choose a random order

    for f in range(partitions):
        # evenly partition the test data

        test_ids = list(np.random.choice(phase_one_ids, size=n_test, replace=False))
        remaining_phase_one = sorted(list(set(phase_one_ids) - set(test_ids)))
        dev_ids = list(np.random.choice(remaining_phase_one, size=n_dev, replace=False))
        train_ids = list(set(item_ids) - set(test_ids) - set(dev_ids))
        print(len(item_ids), len(train_ids), len(dev_ids), len(test_ids))

        for weighting in ['basic', 'label-weights']:
            if weighting == 'label-weights':
                use_label_weights = True
                outdir = os.path.join(basedir, 'label-weights', 'folds',  str(f))
            else:
                use_label_weights = False
                outdir = os.path.join(basedir, 'basic', 'folds',  str(f))

            if not os.path.exists(outdir):
                os.makedirs(outdir)

            write_to_file(outdir, 'train', train_ids, data, item_probs, use_label_weights=use_label_weights, train=True)

            write_to_file(outdir, 'dev', dev_ids, data, item_probs, train=False)

            write_to_file(outdir, 'test', test_ids, data, item_probs, train=False)


    # Finally output everything as training data
    for weighting in ['basic', 'label-weights']:
        if weighting == 'label-weights':
            use_label_weights = True
            outdir = os.path.join(basedir, 'label-weights')
        else:
            use_label_weights = False
            outdir = os.path.join(basedir, 'basic')

        write_to_file(outdir, 'all', item_ids, data, item_probs, use_label_weights=use_label_weights, train=True)


def write_to_file(outdir, output_prefix, item_ids, data, item_probs, use_label_weights=False, train=False):

    inverse_labels = {'no': 'yes', 'yes': 'no'}

    outlines = []
    outlines_json = []
    label_counts = defaultdict(int)

    for item_id in item_ids:
        text = data[item_id]['text']
        text = text.strip()
        sample_prob = data[item_id]['sample_prob']
        phase = data[item_id]['phase']
        yes_prob = item_probs[item_id]
        if yes_prob >= 0.5:
            label = 'yes'
            label_prob = yes_prob
        else:
            label = 'no'
            label_prob = 1. - yes_prob

        if train:
            weight = 1.
            if use_label_weights:
                weight *= label_prob
            outlines.append(text + '\t' + label + '\t' + str(weight) + '\n')
            outlines_json.append({'id': item_id, 'text': text, 'tokens': data[item_id]['tokens'], 'label': label, 'phase': phase, 'weight': weight})
            label_counts[label] += 1

        else:
            weight = 1. / sample_prob
            outlines.append(text + '\t' + label + '\t' + str(weight) + '\n')
            outlines_json.append({'id': item_id, 'text': text, 'tokens': data[item_id]['tokens'], 'label': label, 'phase': phase, 'weight': weight})
            label_counts[label] += 1

    if train:
        random.shuffle(outlines)

    print(label_counts)

    outfile = os.path.join(outdir, output_prefix + '.tsv')
    with open(outfile, 'w') as fo:
        fo.writelines(outlines)

    outfile = os.path.join(outdir, output_prefix + '.ids.json')
    with open(outfile, 'w') as fo:
        json.dump(item_ids, fo)

    outfile = os.path.join(outdir, output_prefix + '.jsonlist')
    with open(outfile, 'w') as fo:
        for line in outlines_json:
            fo.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
