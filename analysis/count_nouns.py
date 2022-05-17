import os
import json
from optparse import OptionParser
from collections import defaultdict, Counter

import pandas as pd
from tqdm import tqdm

from time_periods.common import periods, congress_to_year


# Script to count nouns in the Congressional Record

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='data/speeches/Congress/',
                      help='Base directory: default=%default')
    parser.add_option('--first-congress', type=int, default=46,
                      help='First congress: default=%default')
    parser.add_option('--last-congress', type=int, default=116,
                      help='Last congress: default=%default')
    parser.add_option('--uscr-transition', type=int, default=112,
                      help='Cognress at which to start using USCR dat: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    first_congress = options.first_congress
    last_congress = options.last_congress
    uscr_transition = options.uscr_transition

    hein_dir = os.path.join(basedir, 'hein-bound_parsed')
    uscr_dir = os.path.join(basedir, 'uscr_parsed')
    outdir = os.path.join(basedir, 'basic_counts')
    procedural_file = os.path.join(basedir, 'procedural_speech_ids.txt')

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    congress_range = list(range(first_congress, last_congress+1))
    years = [congress_to_year(c) for c in congress_range]
    print(years[0], years[-1])

    # Load speeches to exclude
    print("Loading procedural speech ids")
    with open(procedural_file) as f:
        to_exclude = f.readlines()
    to_exclude = set([s.strip() for s in to_exclude])
    print(len(to_exclude))

    noun_counter = Counter()
    
    for congress in range(first_congress, last_congress+1):
        print(congress)
        if congress < uscr_transition:
            infile = os.path.join(hein_dir, 'speeches_' + str(congress).zfill(3) + '.jsonlist')
        else:
            infile = os.path.join(uscr_dir, 'speeches_' + str(congress).zfill(3) + '.jsonlist')

        with open(infile) as f:
            for line in tqdm(f):
                line = json.loads(line)
                speech_id = str(line['id'])
                if speech_id != 'speech_id' and speech_id not in to_exclude:
                    sents = line['tokens']
                    tags = line['tags']
                    for s_i, tokens in enumerate(sents):
                        nouns = [token.lower() for t_i, token in enumerate(tokens) if tags[s_i][t_i].startswith('N')]
                        noun_counter.update(nouns)

    for noun, count in noun_counter.most_common(n=20):
        print(noun, count)

    with open(os.path.join(outdir, 'noun_counts.json'), 'w') as f:
        json.dump(noun_counter, f, indent=2)
    

if __name__ == '__main__':
    main()
