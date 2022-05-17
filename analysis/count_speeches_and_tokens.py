import os
import json
from optparse import OptionParser
from collections import defaultdict, Counter

import pandas as pd
from tqdm import tqdm

from time_periods.common import congress_to_year


# Count non-procedural speeches and tokens

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--imm-file', type=str, default='data/speeches/Congress/imm_segments_with_tone_and_metadata.jsonlist',
                      help='Imm segments file: default=%default')
    parser.add_option('--hein-dir', type=str, default='data/speeche/Congress/hein-bound_tokenized/',
                      help='Hein tokenized dir: default=%default')
    parser.add_option('--uscr-dir', type=str, default='data/speeche/Congress/uscr_tokenized/',
                      help='USCR tokenized dir: default=%default')
    parser.add_option('--metadata-dir', type=str, default='data/speeche/Congress/metadata/',
                      help='Metadata directory: default=%default')
    parser.add_option('--procedural-file', type=str, default='data/speeche/Congress/procedural_speech_ids.txt',
                      help='File with list of procdeural speeches: default=%default')
    parser.add_option('--outdir', type=str, default='data/speeche/Congress/basic_counts/',
                      help='Output dir: default=%default')
    parser.add_option('--first-congress', type=int, default=46,
                      help='First congress: default=%default')
    parser.add_option('--last-congress', type=int, default=116,
                      help='Last congress: default=%default')
    parser.add_option('--uscr-transition', type=int, default=112,
                      help='Cognress at which to start using USCR dat: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    imm_file = options.imm_file
    hein_dir = options.hein_dir
    uscr_dir = options.uscr_dir
    metadata_dir = options.metadata_dir
    procedural_file = options.procedural_file
    outdir = options.outdir
    first_congress = options.first_congress
    last_congress = options.last_congress
    uscr_transition = options.uscr_transition

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

    speech_counts_by_congress = Counter()
    speech_counts_by_congress_by_party = defaultdict(Counter)
    token_counts_by_congress = Counter()
    token_counts_by_congress_by_party = defaultdict(Counter)
    imm_token_counts_by_congress = Counter()
    imm_token_counts_by_congress_by_party = defaultdict(Counter)

    for congress in range(first_congress, last_congress+1):
        print(congress)
        if congress < uscr_transition:
            # Note that I saved the parsed files as .txt by mistake; they are actually .jsonlist
            infile = os.path.join(hein_dir, 'speeches_' + str(congress).zfill(3) + '.jsonlist')
            metadata_file = os.path.join(metadata_dir, 'metadata_' + str(congress).zfill(3) + '.json')
        else:
            infile = os.path.join(uscr_dir, 'speeches_' + str(congress).zfill(3) + '.jsonlist')
            metadata_file = os.path.join(metadata_dir, 'uscr_metadata_' + str(congress).zfill(3) + '.json')
        with open(metadata_file) as f:
            metadata = json.load(f)

        with open(infile) as f:
            for line in tqdm(f):
                line = json.loads(line)
                speech_id = str(line['id'])
                sents = line['tokens']
                n_tokens = sum([len(sent) for sent in sents])
                if speech_id in to_exclude:
                    pass
                elif speech_id != 'speech_id':
                    party = metadata[speech_id]['party']
                    speech_counts_by_congress[congress] += 1
                    token_counts_by_congress[congress] += n_tokens
                    if party == 'D' or party == 'R':
                        speech_counts_by_congress_by_party[party][congress] += 1
                        token_counts_by_congress_by_party[party][congress] += n_tokens

    print("Loading imm segments")
    with open(imm_file) as f:
        imm_segments = f.readlines()
    for line in tqdm(imm_segments):
        line = json.loads(line)
        speech_id = str(line['speech_id'])
        if speech_id not in to_exclude:
            congress = int(line['congress'])
            party = line['party']
            text = line['text']
            tokens = text.split()
            n_tokens = len(tokens)
            imm_token_counts_by_congress[congress] += n_tokens
            if party == 'D' or party == 'R':
                imm_token_counts_by_congress_by_party[party][congress] += n_tokens

    with open(os.path.join(outdir, 'speeches_by_congress.json'), 'w') as f:
        json.dump(speech_counts_by_congress, f, indent=2)

    with open(os.path.join(outdir, 'speeches_by_congress_by_party.json'), 'w') as f:
        json.dump(speech_counts_by_congress_by_party, f, indent=2)

    with open(os.path.join(outdir, 'tokens_by_congress.json'), 'w') as f:
        json.dump(token_counts_by_congress, f, indent=2)

    with open(os.path.join(outdir, 'tokens_by_congress_by_party.json'), 'w') as f:
        json.dump(token_counts_by_congress_by_party, f, indent=2)

    with open(os.path.join(outdir, 'imm_tokens_by_congress.json'), 'w') as f:
        json.dump(imm_token_counts_by_congress, f, indent=2)

    with open(os.path.join(outdir, 'imm_tokens_by_congress_by_party.json'), 'w') as f:
        json.dump(imm_token_counts_by_congress_by_party, f, indent=2)


if __name__ == '__main__':
    main()
