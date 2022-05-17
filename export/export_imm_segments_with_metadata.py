import os
import json
from optparse import OptionParser
from collections import defaultdict, Counter

import pandas as pd
from tqdm import tqdm

from metadata.load_metadata import load_metadata, year_to_period


def main():
    usage = "%prog outfile"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='data/speeches/Congress',
                      help='Basedir: default=%default')
    parser.add_option('--first', type=int, default=43,
                      help='First congress: default=%default')
    parser.add_option('--last', type=int, default=114,
                      help='Last congress: default=%default')
    parser.add_option('--chamber', type=str, default=None,
                      help='Only use one chamber (H or S): default=%default')
    parser.add_option('--metadata-early', type=str, default='imm_speeches_with_metadata_early.csv',
                      help='Early metadata file: default=%default')
    parser.add_option('--metadata-mid', type=str, default='imm_speeches_with_metadata_modern_72-86.csv',
                      help='Middle metadata file: default=%default')
    parser.add_option('--metadata-modern', type=str, default='imm_speeches_with_metadata_modern.csv',
                      help='Modern metadata file: default=%default')
    parser.add_option('--states-only', action="store_true", default=False,
                      help='Exclude None states: default=%default')
    parser.add_option('--exclude-exec', action="store_true", default=False,
                      help='Only include Democrat and Republicans with states: default=%default')
    #parser.add_option('--imm-only', action="store_true", default=False,
    #                  help='Only export speeches about immigration: default=%default')

    (options, args) = parser.parse_args()

    outfile = args[0]

    basedir = options.basedir
    hein_bound_dir = basedir + 'hein-bound'
    hein_bound_tokenized_dir = basedir + 'hein-bound_tokenized_rejoined/'
    hein_daily_dir = basedir + 'hein-daily'
    hein_daily_tokenized_dir = basedir + 'hein-daily_tokenized/'
    segments_dir = basedir + 'segments'
    first = options.first
    last = options.last
    target_chamber = options.chamber
    early_file = options.metadata_early
    mid_file = options.metadata_mid
    modern_file = options.metadata_modern
    states_only = options.states_only
    exclude_exec = options.exclude_exec
    imm_only = True

    early_meta = pd.read_csv(os.path.join(basedir, early_file), header=0, index_col=0)
    imm_speech_ids_early = set([str(s) for s in early_meta['speech_id'].values])
    mid_meta = pd.read_csv(os.path.join(basedir, mid_file), header=0, index_col=0)
    imm_speech_ids_mid = set([str(s) for s in mid_meta['speech_id'].values])
    modern_meta = pd.read_csv(os.path.join(basedir, modern_file), header=0, index_col=0)
    imm_speech_ids_modern = set([str(s) for s in modern_meta['speech_id'].values])
    imm_speech_ids = imm_speech_ids_early.union(imm_speech_ids_mid).union(imm_speech_ids_modern)

    metadata = load_metadata(hein_bound_dir, first, last, hein_daily_dir, use_executive_party=False)

    states_to_exclude = {'None', 'DC', 'PI', 'PR', 'GU', 'VI', 'AS', 'MP', 'DK'}

    outlines = []
    for congress in tqdm(range(first, last+1)):
        infile = os.path.join(segments_dir, 'segments-' + str(congress).zfill(3) + '.jsonlist')
        with open(infile) as f:
            for line_i, line in enumerate(f):
                line = json.loads(line)
                segment_id = line['id']
                parts = segment_id.split('_')
                if len(parts) != 2:
                    print('\nskipping', segment_id)
                else:
                    speech_id, chunk = segment_id.split('_')
                    if speech_id != 'speech_id':
                        text = line['text']
                        year, month, day, chamber, name, speaker, lastname, lastname2, firstname, party, inferred_party, gender, gender2, best_state, state, state2, district, nonvoting = metadata[speech_id]

                        skip = False
                        if states_only and best_state in states_to_exclude:
                            skip = True
                        if target_chamber is not None and chamber != target_chamber:
                            skip = True
                        if exclude_exec:
                            if inferred_party != 'D' and inferred_party != 'R':
                                skip = True
                            if best_state in states_to_exclude:
                                skip = True

                        if imm_only and speech_id not in imm_speech_ids:
                            skip = True

                        if not skip:
                            outline = {'id': segment_id, 'text': text, 'year': year, 'state': best_state, 'party': inferred_party, 'chamber': chamber}
                            outlines.append(outline)

    print(len(outlines))
    with open(outfile, 'w') as f:
        for line in outlines:
            f.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
