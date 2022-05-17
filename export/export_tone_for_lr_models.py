import os
import re
import json
import random
from optparse import OptionParser
from collections import defaultdict, Counter

import spacy
import numpy as np
from tqdm import tqdm


# Export the pro and anti segments about immigration as basic (tokenized) text documents
# also export year, decade, congress, state, party, and chamber, as well as the subset of verbs and adjectives


def main():
    usage = "%prog base_outdir"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='data/speeches/Congress',
                      help='Base dir dir: default=%default')
    parser.add_option('--imm-file', type=str, default='data/speeches/Congress/imm_segments_with_tone_and_metadata.jsonlist',
                      help='Immigration file: default=%default')
    parser.add_option('--procedural-file', type=str, default='data/speeches/Congress/procedural_speech_ids.txt',
                      help='File with procedural speeches to exclude: default=%default')
    parser.add_option('--first-congress', type=int, default=46,
                      help='First congress: default=%default')
    parser.add_option('--last-congress', type=int, default=116,
                      help='Last congress: default=%default')
    parser.add_option('--include-neutral', action="store_true", default=False,
                      help='Include neutral speeches: default=%default')

    (options, args) = parser.parse_args()

    outdir = args[0]

    imm_file = options.imm_file
    procedural_file = options.procedural_file
    first_congress = options.first_congress
    last_congress = options.last_congress
    include_neutral = options.include_neutral

    print("Loading spacy")
    nlp = spacy.load("en_core_web_sm")

    with open(imm_file) as f:
        lines = f.readlines()
    lines = [json.loads(line) for line in lines]

    # Load speeches to exclude
    with open(procedural_file) as f:
        to_exclude = f.readlines()
    to_exclude = set([s.strip() for s in to_exclude])
    print(len(to_exclude))

    outlines = []
    tone_map = {0: 'anti', 1: 'neutral', 2: 'pro'}
    for line in tqdm(lines, total=len(lines)):
        speech_id = line['speech_id']
        segment_id = str(line['speech_id']) + '_' + str(line['segment']).zfill(5)
        congress = int(line['congress'])
        pro = float(line['pro'])
        neutral = float(line['neutral'])
        anti = float(line['anti'])
        tone = int(np.argmax([anti, neutral, pro]))
        if first_congress <= congress <= last_congress:
            if speech_id not in to_exclude:
                skip = False
                if tone == 1:
                    if include_neutral:
                        skip = False
                    else:
                        skip = True

                if not skip:
                    parsed = nlp(line['text'])
                    lemmas = []
                    for sent in parsed.sents:
                        lemmas.append([token.lemma_ for token in sent])

                    outlines.append({'id': segment_id, 'tokens': lemmas, 'congress': congress, 'tone': tone_map[tone]})

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    random.shuffle(outlines)
    n_outlines = len(outlines)
    print(n_outlines)

    # save 80% train, 20% dev
    with open(os.path.join(outdir, 'train.tokenized.jsonlist'), 'w') as f:
        for line in outlines[:int(n_outlines*0.9)]:
            f.write(json.dumps(line) + '\n')

    with open(os.path.join(outdir, 'test.tokenized.jsonlist'), 'w') as f:
        for line in outlines[int(n_outlines*0.9):]:
            f.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
