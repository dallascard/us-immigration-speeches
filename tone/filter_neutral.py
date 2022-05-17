import os
import json
import random
from optparse import OptionParser
from collections import defaultdict




def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--infile', type=str, default='data/speeches/Congress/tone/splits/all/all.jsonlist',
                      help='Infile: default=%default')

    (options, args) = parser.parse_args()

    infile = options.infile
    outdir, filename = os.path.split(infile)

    with open(infile) as f:
        lines = f.readlines()
    lines = [json.loads(line) for line in lines]
    print(len(lines))

    outlines = [line for line in lines if line['label'] != 'neutral']
    print(len(outlines))

    outfile = os.path.join(outdir, infile[:-9] + '_minus_neutral.jsonlist')
    with open(outfile, 'w') as f:
        for line in outlines:
            f.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
