import os
import json
import random
from optparse import OptionParser
from collections import defaultdict, Counter

import numpy as np

from time_periods.common import congress_to_decade

# Combine all the labeled relevance segments and output some basic data splits


def main():
    usage = "%prog "
    parser = OptionParser(usage=usage)
    parser.add_option('--indir', type=str, default='data/speeches/Congress/data/annotations/relevance_and_tone/inferred_labels/',
                      help='Dir with inferred labels: default=%default')
    parser.add_option('--test', type=int, default=500,
                      help='Number of test examples: default=%default')
    parser.add_option('--seed', type=int, default=42,
                      help='Random seed: default=%default')
    parser.add_option('--outdir', type=str, default='data/speeches/Congress/predictions_val/relevance/',
                      help='Base output directory: default=%default')

    (options, args) = parser.parse_args()

    indir = options.indir
    seed = options.seed
    test = options.test
    outdir = options.outdir

    all_lines = []
    
    with open (os.path.join(indir, 'early_relevance_all.jsonlist')) as f:
        lines = f.readlines()
        print(len(lines))
        all_lines.extend(lines)

    with open (os.path.join(indir, 'mid_relevance_all.jsonlist')) as f:
        lines = f.readlines()
        print(len(lines))
        all_lines.extend(lines)

    with open (os.path.join(indir, 'modern_relevance_all.jsonlist')) as f:
        lines = f.readlines()
        print(len(lines))
        all_lines.extend(lines)

    print(len(all_lines))

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

    for d in sorted(decade_counter):
        print(d, decade_counter[d])

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with open(os.path.join(outdir, 'all.jsonlist'), 'w') as f:
        for line in outlines:
            f.write(json.dumps(line) + '\n') 

    np.random.seed(options.seed)
    random.seed(options.seed)

    test_indices = np.random.choice(np.arange(len(outlines)), size=test, replace=False)
    train_indices = sorted(set(np.arange(len(outlines))) - set(test_indices))
    np.random.shuffle(train_indices)

    with open(os.path.join(outdir, 'train.jsonlist'), 'w') as f:
        for i in train_indices:
            f.write(json.dumps(outlines[i]) + '\n') 
    
    with open(os.path.join(outdir, 'test.jsonlist'), 'w') as f:
        for i in test_indices:
            f.write(json.dumps(outlines[i]) + '\n') 


if __name__ == '__main__':
    main()
