import os
import json
from optparse import OptionParser
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

def main():
    usage = "%prog model_dir test.jsonlist"
    parser = OptionParser(usage=usage)
    #parser.add_option('--model-dir', type=str, default='',
    #                  help='Part of data to predict on [train|dev|test]: default=%default')
    #parser.add_option('--eval', action="store_true", default=False,
    #                  help='Evaluate predictions: default=%default')

    (options, args) = parser.parse_args()
    model_dir = args[0]
    test_file = args[1]

    with open(test_file) as f:
        lines = f.readlines()
    lines = [json.loads(line) for line in lines]

    pred_file = os.path.join(model_dir, 'pred.probs.test.csv')
    df = pd.read_csv(pred_file, header=0, index_col=0)

    label_list = list(df.columns)
    label_index = dict(zip(label_list, np.arange(len(label_list))))
    print(label_list)
    preds = df.values
    pred_labels = np.argmax(preds, 1)

    early_last = 73
    modern_first = 85

    n_labels = len(label_list)
    cfm = np.zeros([n_labels, n_labels])
    cfm_early = np.zeros([n_labels, n_labels])
    cfm_mid = np.zeros([n_labels, n_labels])
    cfm_modern = np.zeros([n_labels, n_labels])

    for i, line in enumerate(lines):
        true = label_index[line['label']]
        pred = int(pred_labels[i])
        cfm[true, pred] += 1
        speech_id = line['id']
        if speech_id.startswith('1'):
            congress = int(line['id'][:3])
        else:
            congress = int(line['id'][:2])
        if congress <= early_last:
            cfm_early[true, pred] += 1
        elif congress < modern_first:
            cfm_mid[true, pred] += 1
        else:
            cfm_modern[true, pred] += 1

    print(cfm)

    cfm_norm = cfm / cfm.sum(1)
    print(cfm_norm)

    print('\nEarly:')    
    print(cfm_early / cfm_early.sum(1))

    print('\nMid:')    
    print(cfm_mid / cfm_mid.sum(1))

    print('\nModern:')    
    print(cfm_modern / cfm_modern.sum(1))


if __name__ == '__main__':
    main()
