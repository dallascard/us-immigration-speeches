import os
import glob
import json
from itertools import combinations
from collections import defaultdict
from optparse import OptionParser

import numpy as np
import pandas as pd


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--indir', type=str, default='data/speeches/Congressfor_annotation_mid/',
                      help='Input dir: default=%default')
    parser.add_option('--basedir', type=str, default='data/speeches/Congress/for_annotation_mid/rounds/',
                      help='Base output dir: default=%default')
    parser.add_option('--annotators', type=int, default=4,
                      help='Number of annotators: default=%default')
    parser.add_option('--per-item', type=int, default=2,
                      help='Annotations per item: default=%default')
    parser.add_option('--per-batch', type=int, default=120,
                      help='Number of items per batch: default=%default')
    parser.add_option('--start', type=int, default=0,
                      help='Start index: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    indir = options.indir
    basedir = options.basedir
    n_annotators = options.annotators
    n_annotators_per_item = options.per_item
    items_per_batch = options.per_batch
    start = options.start

    files = glob.glob(indir + '*.jsonlist')
    files.sort()

    np.random.seed(0)

    all_lines = []
    for file_i, infile in enumerate(files):
        with open(infile) as f:
            lines = f.readlines()
        lines = [json.loads(line) for line in lines]
        all_lines.extend(lines)

    print(len(all_lines))

    order = np.arange(len(all_lines))
    np.random.shuffle(order)

    combs = []
    for comb in combinations(np.arange(n_annotators), n_annotators_per_item):
        combs.append(list(comb))
        print(comb)
    n_combs = len(combs)
    print(n_combs)

    total_items = items_per_batch * n_annotators // n_annotators_per_item

    outlines_by_annotator = defaultdict(list)

    for i in range(total_items):
        index = order[start+i]
        line = all_lines[index]
        segment_id = line['id']
        congress = line['infile'][9:12]
        year = 1873 + (int(congress) - 43) * 2
        text = line['text']
        outline = [segment_id, congress, year, text]
        annotators = combs[i % n_combs]
        for a in annotators:
            outlines_by_annotator[a].append(outline)

    for a, items in outlines_by_annotator.items():
        print(a, len(items))

    outdir = os.path.join(basedir, str(start))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for a in range(n_annotators):
        df = pd.DataFrame(outlines_by_annotator[a], columns=['id', 'congress', 'year', 'text'])
        df.to_csv(os.path.join(outdir, str(a+1) + '.csv'))

    print("Next start:", start + total_items)
    with open(os.path.join(outdir, 'next_start.txt'), 'w') as f:
        f.write('Next start: ' + str(start+total_items))


if __name__ == '__main__':
    main()
