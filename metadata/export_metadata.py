import os
import json
from optparse import OptionParser

import pandas as pd

from metadata.load_metadata import load_metadata


def main():
    usage = "%prog outdir"
    parser = OptionParser(usage=usage)
    parser.add_option('--hein-bound-dir', type=str, default='data/speeches/Congress/hein-bound/',
                      help='Issue: default=%default')
    parser.add_option('--hein-daily-dir', type=str, default='data/speeches/Congress/hein-daily/',
                      help='Issue: default=%default')
    parser.add_option('--first', type=int, default=43,
                      help='First congress: default=%default')
    parser.add_option('--last', type=int, default=71,
                      help='Last congress: default=%default')

    (options, args) = parser.parse_args()

    outdir = args[0]

    hein_bound_dir = options.hein_bound_dir
    hein_daily_dir = options.hein_daily_dir
    first = options.first
    last = options.last

    metadata = load_metadata(hein_bound_dir, first, last, hein_daily_dir)

    columns = ['year', 'month', 'day', 'chamber', 'name', 'speaker', 'lastname', 'lastname2', 'firstname', 'party', 'inferred_party', 'gender', 'gender2', 'best_state', 'state', 'state2', 'district', 'nonvoting']
    df = pd.DataFrame()

    speech_ids = sorted(metadata)
    for c_i, column in enumerate(columns):
        vals = [metadata[speech_id][c_i] for speech_id in speech_ids]
        df[column] = vals
    metadata_subset = []
    for speech_id, data in metadata.items():
        year, month, day, chamber, name, speaker, lastname, lastname2, firstname, party, inferred_party, gender, gender2, best_state, state, state2, district, nonvoting = data
        metadata_subset.append({'speech_id': speech_id, 'year': year, 'month': month, 'day': day, 'chamber': chamber, 'name': name, 'inferred_party': inferred_party, 'party': party, 'state': best_state})

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    df.to_csv(os.path.join(outdir, 'metadata_{:d}-{:d}.csv'.format(first, last)))

    with open(os.path.join(outdir, 'metadata_{:d}-{:d}.jsonlist'.format(first, last)), 'w') as f:
        for line in metadata_subset:
            f.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
