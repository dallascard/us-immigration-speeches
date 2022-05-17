import os
import re
import json
from optparse import OptionParser
from collections import defaultdict, Counter

from tqdm import tqdm


def main():
    usage = "%prog outfile.json"
    parser = OptionParser(usage=usage)
    parser.add_option('--indir', type=str, default='data/speeches/Congress/uscr/',
                      help='Issue: default=%default')
    parser.add_option('--first', type=int, default=113,
                      help='First congress: default=%default')
    parser.add_option('--last', type=int, default=116,
                      help='Last congress: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    outfile = args[0]

    indir = options.indir
    first = options.first
    last = options.last

    lengths_by_congress = defaultdict(list)
    for congress in tqdm(range(first, last+1)):
        infile = os.path.join(indir, 'speeches_' + str(congress).zfill(3) + '.jsonlist')

        with open(infile) as f:
            lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            text = line['text']
            year = int(line['year'])
            lengths_by_congress[year].append(len(text.split()))

    with open(outfile, 'w') as f:
        json.dump(lengths_by_congress, f)


if __name__ == '__main__':
    main()


