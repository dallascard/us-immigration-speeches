import os
import json
from optparse import OptionParser
from collections import defaultdict, Counter


def main():
    usage = "%prog outfile.json"
    parser = OptionParser(usage=usage)
    parser.add_option('--uscr-dir', type=str, default='data/speeches/Congress/uscr/',
                      help='USCR speech dir: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    outfile = args[0]

    uscr_dir = options.uscr_dir

    first = 104
    last = 116

    speech_dates = {}
    for congress in range(first, last+1):
        print(congress)
        infile = os.path.join(uscr_dir, 'speeches_' + str(congress).zfill(3) + '.jsonlist')
        with open(infile) as f:
            lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            speech_id = line['id']
            date = line['date']
            assert len(date) == 8
            speech_dates[speech_id] = date

    with open(outfile, 'w') as fo:
        json.dump(speech_dates, fo, indent=2)


if __name__ == '__main__':
    main()
