import os
import json
from optparse import OptionParser
from collections import defaultdict, Counter


# Split a jsonlist file into pieces, so as to be able to process on GPU

def main():
    usage = "%prog infile.jsonlist"
    parser = OptionParser(usage=usage)
    parser.add_option('--pieces', type=int, default=5,
                      help='Number of pieces to break the file into: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    infile = args[0]

    n_pieces = options.pieces

    print("Loading file")
    with open(infile) as f:
        lines = f.readlines()

    size = len(lines) // n_pieces

    for p in range(n_pieces):
        print(p)
        if p == n_pieces-1:
            outlines = lines[p * size:]
        else:
            outlines = lines[p * size: (p+1) * size]
        outfile = infile[:-9] + '_' + str(p) + '.jsonlist'
        print(outfile)
        with open(outfile, 'w') as f:
            for line in outlines:
                f.write(line)


if __name__ == '__main__':
    main()
