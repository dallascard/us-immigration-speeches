import os
import re
import json
from glob import glob
from collections import Counter, defaultdict
from optparse import OptionParser

import numpy as np
import pandas as pd
from tqdm import tqdm


# Export the tokens, lemmas, and tags from the parsed data for those speeches that are about immigration
# while accounting for a couple of things that I have taken care of elsewhere for other formats


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--imm-file', type=str, default='data/speeches/Congress/imm_speech_ids_with_tone.tsv',
                      help='.tsv file with immigration segments and tones (from tone.collect_predictions.py): default=%default')
    parser.add_option('--hein-dir', type=str, default='data/speeches/Congress/hein-bound_tokenized/',
                      help='Tokenized hein bound dir: default=%default')
    parser.add_option('--uscr-dir', type=str, default='data/speeches/Congress/uscr_tokenized/',
                      help='Tokenized hein daily dir: default=%default')
    parser.add_option('--first', type=int, default=46,
                      help='First congress: default=%default')
    parser.add_option('--last', type=int, default=116,
                      help='Last congress: default=%default')
    parser.add_option('--uscr-transition', type=int, default=112,
                      help='Congress at which to switch to USCR data: default=%default')
    parser.add_option('--outfile', type=str, default='data/speeches/Congress/imm_speeches.jsonlist',
                      help='Outfile: default=%default')

    (options, args) = parser.parse_args()

    imm_file = options.imm_file
    hein_bound_dir = options.hein_dir
    uscr_dir = options.uscr_dir
    outfile = options.outfile
    first = options.first
    last = options.last
    uscr_transition = options.uscr_transition

    print("Loading imm speech ids")
    df = pd.read_csv(imm_file, sep='\t', header=0, index_col=None)
    imm_speech_ids = set([str(i) for i in df['speech_id'].values])

    outlines = []

    for congress in range(first, last+1):
        print(congress)
        if congress < uscr_transition:
            infile = os.path.join(hein_bound_dir, 'speeches_' + str(congress).zfill(3) + '.jsonlist')
        else:
            infile = os.path.join(uscr_dir, 'speeches_' + str(congress).zfill(3) + '.jsonlist')

        with open(infile) as f:
            lines = f.readlines()

        for line in tqdm(lines):
            line = json.loads(line)
            speech_id = line['id']
            if speech_id in imm_speech_ids:
                sent_tokens = line['tokens']
                text = ''
                for tokens in sent_tokens:
                    text += ' '.join(tokens) + ' '
                outlines.append({'id': speech_id, 'text': text.strip(), 'congress': congress})

    with open(outfile, 'w') as f:
        for line in outlines:
            f.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
