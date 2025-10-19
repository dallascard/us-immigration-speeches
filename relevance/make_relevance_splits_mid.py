import os
import json
import random
from optparse import OptionParser
from collections import defaultdict

import numpy as np

from relevance.make_relevance_splits_modern import write_to_file

# Script to take the output of the label aggregation model and create multiple folds of data for training and evaluation


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--text-file', type=str, default='texts.json',
                      help='Output of annotations.tokenize: default=%default')
    parser.add_option('--probs-file', type=str, default='item_probs.json',
                      help='Output of label-aggregation: default=%default')
    parser.add_option('--seed', type=int, default=42,
                      help='Random seed: default=%default')
    parser.add_option('--outdir', type=str, default='data/mid/splits/',
                      help='Base output directory: default=%default')

    (options, args) = parser.parse_args()

    data_file = options.text_file
    item_probs_file = options.probs_file     # estimated label probabilities from label aggregation
    basedir = options.outdir
    np.random.seed(options.seed)
    random.seed(options.seed)

    # Load the annotations (one annotation per pine)
    print("Loading data")
    with open(data_file) as f:
        data = json.load(f)
    print(len(data))

    # load the item probs
    with open(item_probs_file) as f:
        item_probs = json.load(f)
    print(len(item_probs))

    assert len(data) >= len(item_probs)

    print(len(data))
    print(len(item_probs))

    item_ids = sorted(item_probs)

    # choose a random order

    item_ids = sorted(item_probs)

    # Finally output everything as training data
    for weighting in ['basic', 'label-weights']:

        if weighting == 'label-weights':
            use_label_weights = True
            outdir = os.path.join(basedir, 'label-weights')
        else:
            use_label_weights = False
            outdir = os.path.join(basedir, 'basic')

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        write_to_file(outdir, 'all', item_ids, data, item_probs, use_label_weights=use_label_weights, train=True, extra_train_lines=None)

if __name__ == '__main__':
    main()
