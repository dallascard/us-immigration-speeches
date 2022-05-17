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
    parser.add_option('--hein-parsed-dir', type=str, default='data/speeches/Congress/hein-bound_parsed/',
                      help='Parsed hein bound dir: default=%default')
    parser.add_option('--uscr-parsed-dir', type=str, default='data/speeches/Congress/uscr_parsed/',
                      help='Parsed hein daily dir: default=%default')
    parser.add_option('--metadata-dir', type=str, default='data/speeches/Congress/metadata/',
                      help='Metadata dir: default=%default')
    parser.add_option('--first', type=int, default=43,
                      help='First congress: default=%default')
    parser.add_option('--last', type=int, default=116,
                      help='Last congress: default=%default')
    parser.add_option('--uscr-transition', type=int, default=112,
                      help='Congress at which to switch to USCR data: default=%default')
    parser.add_option('--outfile', type=str, default='data/speeches/Congress/imm_speeches_parsed.jsonlist',
                      help='Model name or path: default=%default')

    (options, args) = parser.parse_args()

    imm_file = options.imm_file
    hein_bound_dir = options.hein_parsed_dir
    uscr_dir = options.uscr_parsed_dir
    metadata_dir = options.metadata_dir
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
            infile = os.path.join(hein_bound_dir, 'speeches_' + str(congress).zfill(3) + '.txt')
            metadata_file = os.path.join(metadata_dir, 'metadata_' + str(congress).zfill(3) + '.json')
        else:
            infile = os.path.join(uscr_dir, 'speeches_' + str(congress).zfill(3) + '.jsonlist')
            metadata_file = os.path.join(metadata_dir, 'uscr_metadata_' + str(congress).zfill(3) + '.json')
        with open(metadata_file) as f:
            metadata = json.load(f)

        with open(infile) as f:
            lines = f.readlines()

        for line in tqdm(lines):
            line = json.loads(line)
            speech_id = line['id']
            if speech_id in imm_speech_ids:
                sent_tokens = line['tokens']
                sent_lemmas = line['lemmas']
                sent_tags = line['tags']
                date = metadata[speech_id]['date']

                if congress < uscr_transition:
                    # Try to fix up sentences which have been oversplit due to commas appearing as periods
                    rejoined_tokens = []
                    rejoined_lemmas = []
                    rejoined_tags = []
                    if len(sent_tokens) > 0:
                        current_tokens = sent_tokens[0]
                        current_lemmas = sent_lemmas[0]
                        current_tags = sent_tags[0]
                        if len(sent_tokens) > 1:
                            for sent_i in range(1, len(sent_tokens)):
                                # look to see if this might be a false sentence break
                                if sent_tokens[sent_i-1][-1] == '.' and (sent_tokens[sent_i][0].islower() or sent_tokens[sent_i][0].isdigit() or sent_tokens[sent_i][0] == '$' or sent_tokens[sent_i][0] == '%'):
                                    # if so, extend the previous sentence / tokens
                                    current_tokens.extend(sent_tokens[sent_i])
                                    current_lemmas.extend(sent_lemmas[sent_i])
                                    current_tags.extend(sent_tags[sent_i])
                                else:
                                    # otherwise, add the previous to the list, and start a new one
                                    rejoined_tokens.append(current_tokens)
                                    rejoined_lemmas.append(current_lemmas)
                                    rejoined_tags.append(current_tags)
                                    current_tokens = sent_tokens[sent_i]
                                    current_lemmas = sent_lemmas[sent_i]
                                    current_tags = sent_tags[sent_i]
                        # add the current to the list
                        rejoined_tokens.append(current_tokens)
                        rejoined_lemmas.append(current_lemmas)
                        rejoined_tags.append(current_tags)
                    sent_tokens = rejoined_tokens
                    sent_lemmas = rejoined_lemmas
                    sent_tags = rejoined_tags

                # one bad day in dataset; should already be excluded, but just in case
                if date != '18940614':
                    outlines.append({'infile': os.path.basename(infile), 'id': speech_id, 'tokens': sent_tokens, 'lemmas': sent_lemmas, 'tags': sent_tags, 'date': date})

    with open(outfile, 'w') as f:
        for line in outlines:
            f.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
