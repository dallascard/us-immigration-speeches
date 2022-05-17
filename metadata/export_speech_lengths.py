import os
import re
import json
from optparse import OptionParser
from collections import defaultdict, Counter

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

    lengths_by_congress = defaultdict(list)
    for congress in tqdm(range(first, last+1)):
        if congress < 112:
            indir = hein_bound_dir
        else:
            indir = hein_daily_dir

        infile = os.path.join(indir, 'speeches_' + str(congress).zfill(3) + '.txt')
        descr_file = os.path.join(indir, 'descr_' + str(congress).zfill(3) + '.txt')

        # then load the speech descriptions (less reliable but more common)
        speech_dates = {}
        with open(descr_file, encoding='Windows-1252') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.split('|')
            speech_id, chamber, date, line_num, speaker, first_name, last_name, state, gender, _, _, _, _, _ = parts
            # exclude a possible header row and one day with corrupted data
            if speech_id != 'speech_id':
                speech_dates[speech_id] = date

        with open(infile, encoding='Windows-1252') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.split('|')
            speech_id = parts[0]
            if speech_id != 'speech_id':
                text = ' '.join(parts[1:])
                date = speech_dates[speech_id]
                if date != '18940614':
                    year = int(date[:4])
                    lengths_by_congress[year].append(len(text.split()))

    with open(outfile, 'w') as f:
        json.dump(lengths_by_congress, f)


if __name__ == '__main__':
    main()


