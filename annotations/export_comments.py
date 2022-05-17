import os
import json
from glob import glob
from optparse import OptionParser
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


def main():
    usage = "%prog basedir outdir"
    parser = OptionParser(usage=usage)
    #parser.add_option('--issue', type=str, default='immigration',
    #                  help='Issue: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    basedir = args[0]
    outdir = args[1]
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    topic_comments = []
    char_comments = []
    files = sorted(glob(os.path.join(basedir, 'round*', '*.tsv')))
    print(len(files))
    for infile in files:
        df = pd.read_csv(infile, header=0, index_col=None, sep='\t')
        topic_comments.extend(df['Topic with respect to immigration (one or two words)'].values)
        char_comments.extend(df['Characterization of immigrants'])

    topic_comments = [c for c in topic_comments if type(c) == str]
    char_comments = [c for c in char_comments if type(c) == str]

    print(len(topic_comments), len(char_comments))

    with open(os.path.join(outdir, 'topic_comments.txt'), 'w') as f:
        for line in topic_comments:
            f.write(line + '\n')

    with open(os.path.join(outdir, 'char_comments.txt'), 'w') as f:
        for line in char_comments:
            f.write(line + '\n')


if __name__ == '__main__':
    main()
