import os
import json
import random
from optparse import OptionParser
from collections import defaultdict, Counter

import spacy
import numpy as np

from time_periods.common import congress_to_decade


# Combine all the labeled tone segments and output some basic data splits


def main():
    usage = "%prog "
    parser = OptionParser(usage=usage)
    parser.add_option('--indir', type=str, default='data/annotations/relevance_and_tone/inferred_labels/',
                      help='Dir with inferred labels: default=%default')
    parser.add_option('--test', type=int, default=400,
                      help='Number of test examples: default=%default')
    parser.add_option('--seed', type=int, default=42,
                      help='Random seed: default=%default')
    parser.add_option('--outdir', type=str, default='data/speeches/Congress/data/tone_combined/',
                      help='Base output directory: default=%default')
    parser.add_option('--subset', type=str, default=None,
                      help='subset to use [early|mid|modern] or None for all: default=%default')
    parser.add_option('--binary', action="store_true", default=False,
                      help='Use only binary tone labels: default=%default')

    (options, args) = parser.parse_args()

    indir = options.indir
    seed = options.seed
    test = options.test
    outdir = options.outdir
    subset = options.subset
    binary = options.binary

    np.random.seed(seed)

    all_lines = []
    
    if subset is None or subset == 'early':
        with open (os.path.join(indir, 'early_tone_all.jsonlist')) as f:
            lines = f.readlines()
        print(len(lines))
        all_lines.extend(lines)

    if subset is None or subset == 'mid':
        with open (os.path.join(indir, 'mid_tone_all.jsonlist')) as f:
            lines = f.readlines()
        print(len(lines))
        all_lines.extend(lines)

    if subset is None or subset == 'modern':
        with open (os.path.join(indir, 'modern_tone_all.jsonlist')) as f:
            lines = f.readlines()
        print(len(lines))
        all_lines.extend(lines)

    print(len(all_lines))

    if subset is None or subset == 'modern':
        with open (os.path.join(indir, 'immigration_primary_tone_train_tokenized.jsonlist')) as f:
            lines = f.readlines()        
        mfc_tone_lines_train = []
        for line in lines:
            line = json.loads(line)
            line['weight'] = 1.0
            mfc_tone_lines_train.append(line)    
        print(len(mfc_tone_lines_train))
        
        with open (os.path.join(indir, 'immigration_primary_tone_test_tokenized.jsonlist')) as f:
            lines = f.readlines()
        mfc_tone_lines_test = []
        for line in lines:
            line = json.loads(line)
            line['weight'] = 1.0
            mfc_tone_lines_test.append(line)
        print(len(mfc_tone_lines_test))

        if binary:
            mfc_tone_lines_train = [line for line in mfc_tone_lines_train if line['label'] != 'neutral']
            mfc_tone_lines_test = [line for line in mfc_tone_lines_test if line['label'] != 'neutral']

    outlines = []
    # add decade information
    decade_counter = Counter()
    for line in all_lines:
        line = json.loads(line)
        line_id = line['id']
        if line_id.startswith('1'):
            congress = int(line_id[:3])
        else:
            congress = int(line_id[:2])
        decade = congress_to_decade(congress) - 5
        decade_counter[int(decade)] += 1
        line['decade'] = str(decade)
        outlines.append(line)

    if binary:
        outlines = [line for line in outlines if line['label'] != 'neutral']

    for d in sorted(decade_counter):
        print(d, decade_counter[d])

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    test_indices = np.random.choice(np.arange(len(outlines)), size=test, replace=False)
    train_indices = sorted(set(np.arange(len(outlines))) - set(test_indices))
    np.random.shuffle(train_indices)

    if subset is None or subset == 'modern':
        train_lines = [outlines[i] for i in train_indices] + mfc_tone_lines_train
    else:
        train_lines = [outlines[i] for i in train_indices]
    test_lines = [outlines[i] for i in test_indices]

    if subset is None or subset == 'modern':    
        all_lines = outlines + mfc_tone_lines_train + mfc_tone_lines_test

    print(len(train_lines))
    print(len(test_lines))

    np.random.shuffle(train_lines)
    np.random.shuffle(all_lines)

    print("Including MFC data")

    if subset is None:
        prefix = ''
    elif subset == 'early':
        prefix = 'early_'
    elif subset == 'mid':
        prefix = 'mid_'
    elif subset == 'modern':
        prefix = 'modern_'
    
    with open(os.path.join(outdir, prefix + 'train.jsonlist'), 'w') as f:
        for line in train_lines:
            f.write(json.dumps(line) + '\n') 
    
    with open(os.path.join(outdir, prefix + 'test.jsonlist'), 'w') as f:
        for line in test_lines:
            f.write(json.dumps(line) + '\n') 

    with open(os.path.join(outdir, prefix + 'all.jsonlist'), 'w') as f:
        for line in all_lines:
            f.write(json.dumps(line) + '\n') 

    if subset is None or subset == 'modern':
        with open(os.path.join(outdir, 'mfc_test.jsonlist'), 'w') as f:
            for line in mfc_tone_lines_test:
                f.write(json.dumps(line) + '\n') 



if __name__ == '__main__':
    main()
