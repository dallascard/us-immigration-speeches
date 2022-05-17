import os
import re
import json
import random
from optparse import OptionParser
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm

from common.functions import simplify_text


# Partial implementation of process to train a classifier to identify proecdural speeches
# This part so far just exports the very short speeches (less than 16 characters or 3 tokens after cleaning)
# the rest is in the notebook "1 identify speeches to exclude"


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--hein-dir', type=str, default='data/speeches/Congress/hein-bound_tokenized/',
                      help='Hein bound tokenized dir: default=%default')
    parser.add_option('--uscr-dir', type=str, default='data/speeches/Congress/uscr_tokenized/',
                      help='USCR tokenized dir: default=%default')
    parser.add_option('--outdir', type=str, default='data/speeches/Congress/procedural/',
                      help='Output dir: default=%default')
    parser.add_option('--uscr-transition', type=int, default=112,
                      help='Congress at which to start using USCR data: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    hein_dir = options.hein_dir
    uscr_dir = options.uscr_dir
    outdir = options.outdir
    uscr_transition = options.uscr_transition
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # random seed for reproducibility
    np.random.seed(84684)

    short_speeches = Counter()
    short_speech_ids = set()
    speech_counter = Counter()
    id_count = 0
    train_lines = []
    test_lines = []

    # go through each congress
    for congress in tqdm(range(43, 117)):
        if congress < uscr_transition:
            infile = os.path.join(hein_dir, 'speeches_' + str(congress).zfill(3) + '.jsonlist')
        else:
            infile = os.path.join(uscr_dir, 'speeches_' + str(congress).zfill(3) + '.jsonlist')
        with open(infile) as f:
            lines = f.readlines()
        lines = [json.loads(line) for line in lines]
        # go through each speech
        long_pieces = []
        for line in lines:
            speech_id = line['id']
            tokens = []
            for sent in line['tokens']:
                tokens.extend(sent)
            text = ' '.join(tokens)
            # drop punctuation and convert to lower case
            text = simplify_text(text)
            tokens = text.split()
            # put very short speeches in one bucket
            if len(tokens) <= 2 or len(text) <= 15:
                short_speech_ids.add(speech_id)
                short_speeches[text] += 1
            # put other short speeches in a counter
            elif len(text) < 200:
                speech_counter[text] += 1
            # also collect the long speeches, and break them into pieces
            elif len(text) > 1000 and len(tokens) > 20:
                # compute how many pieces to use
                n_pieces = (len(text) // 200) + 1
                n_tokens = len(tokens)
                n_tokens_per_piece = n_tokens // n_pieces
                # put each piece into a separate json object, labeled as not procedural
                for n in range(n_pieces):
                    if n < n_tokens_per_piece-1:
                        subset = tokens[n * n_tokens_per_piece: (n+1) * n_tokens_per_piece]
                    else:
                        subset = tokens[n * n_tokens_per_piece:]
                    segment_length = len(' '.join(subset))
                    if segment_length < 1000:
                        outline = {'id': 'x' + str(id_count).zfill(8), 'tokens': [subset], 'procedural': 'no'}
                        long_pieces.append(outline)
                        id_count += 1

        # randomly assign non-procedural speeches to train or test
        # down-sample non-procedural speech pieces by a factor of 50 (heuristically chosen to balance)
        random_indices = np.random.choice(np.arange(len(long_pieces)), size=len(long_pieces)//50, replace=False)
        # select a random test set of 5% of pieces
        test_indices = np.random.choice(random_indices, size=len(random_indices)//20, replace=False)
        # use the remainder as train
        train_indices = list(set(random_indices) - set(test_indices))
        for index in train_indices:
            train_lines.append(long_pieces[index])
        for index in test_indices:
            test_lines.append(long_pieces[index])

    print("Saving {:d} very short speech ids".format(len(short_speech_ids)))
    # save the IDs of the very short speeches
    with open(os.path.join(outdir, 'very_short_speech_ids.txt'), 'w') as f:
        for speech_id in tqdm(sorted(short_speech_ids)):
            f.write(str(speech_id) + '\n')

    # assign 1/10th of the short speeches into train and test, assigned to the procedural = yes category
    n = 0
    for text, count in tqdm(speech_counter.items()):
        # put those that only occur between 10 and 20 times into test data (once)
        if 10 <= count < 20:
            outline = {'id': 'se' + str(n).zfill(8), 'tokens': [text.split()], 'procedural': 'yes'}
            test_lines.append(outline)
            n += 1
        # put those that occur at least 20 times into train (duplicated by a factor of 1/10)
        elif count >= 20:
            for c in range(count//10):
                outline = {'id': 'st' + str(n).zfill(8), 'tokens': [text.split()], 'procedural': 'yes'}
                train_lines.append(outline)
                n += 1

    print("Saving train and test lines")
    print(len(train_lines), len(test_lines))

    # randomly shuffle the train and test set and write them to files
    outfile = os.path.join(outdir, 'short_vs_long_train.jsonlist')
    random.shuffle(train_lines)
    with open(outfile, 'w') as f:
        for line in train_lines:
            f.write(json.dumps(line) + '\n')
    outfile = os.path.join(outdir, 'short_vs_long_test.jsonlist')
    random.shuffle(test_lines)
    with open(outfile, 'w') as f:
        for line in test_lines:
            f.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
