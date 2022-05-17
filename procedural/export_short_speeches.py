import os
import re
import json
import random
from optparse import OptionParser
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm

from common.functions import simplify_text


# Export all shortish (but not very short speeches) for prediction


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

    (options, args) = parser.parse_args()

    hein_dir = options.hein_dir
    uscr_dir = options.uscr_dir
    outdir = options.outdir
    uscr_transition = options.uscr_transition
    if not os.path.exists(outdir):
        os.makedirs(outdir)

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
        shortish_speeches = []
        for line in lines:
            speech_id = line['id']
            tokens = []
            for sent in line['tokens']:
                tokens.extend(sent)
            text = ' '.join(tokens)
            # drop punctuation and convert to lower case
            text = simplify_text(text)
            tokens = text.split()
            # skip very short speeches altogether
            if len(tokens) <= 2 or len(text) <= 15:
                pass
            # export all shortish speeches
            elif len(text) < 400:
                shortish_speeches.append({'id': speech_id, 'tokens': [tokens]})

        with open(os.path.join(outdir, 'short_speeches_' + str(congress).zfill(3) + '.jsonlist'), 'w') as f:
            for line in shortish_speeches:
                f.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
