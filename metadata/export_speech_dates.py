import os
import json
from optparse import OptionParser

from tqdm import tqdm


def main():
    usage = "%prog outfile.json"
    parser = OptionParser(usage=usage)
    parser.add_option('--hein-bound-dir', type=str, default='data/speeches/Congress/hein-bound/',
                      help='Issue: default=%default')
    parser.add_option('--hein-daily-dir', type=str, default='data/speeches/Congress/hein-daily/',
                      help='Issue: default=%default')
    parser.add_option('--first', type=int, default=43,
                      help='First congress: default=%default')
    parser.add_option('--last', type=int, default=114,
                      help='Last congress: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    outfile = args[0]

    hein_bound_dir = options.hein_bound_dir
    hein_daily_dir = options.hein_daily_dir
    first = options.first
    last = options.last

    speech_dates = {}
    for congress in tqdm(range(first, last+1)):
        if congress < 112:
            infile = os.path.join(hein_bound_dir, 'descr_' + str(congress).zfill(3) + '.txt')
        else:
            infile = os.path.join(hein_daily_dir, 'descr_' + str(congress).zfill(3) + '.txt')
        with open(infile, encoding='Windows-1252') as f:
            lines = f.readlines()
        for line in lines[1:]:
            parts = line.split('|')
            speech_id = parts[0]
            date = parts[2]
            assert len(date) == 8
            speech_dates[speech_id] = date

    with open(outfile, 'w') as fo:
        json.dump(speech_dates, fo, indent=2, sort_keys=False)


if __name__ == '__main__':
    main()
