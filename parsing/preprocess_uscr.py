import os
import re
import json
from glob import glob
from optparse import OptionParser
from collections import defaultdict, Counter

from parsing.common import normalize_to_stanford


# Make a few modification to USCR files to make it more like the Gentzkow data

def main():
    usage = "%prog uscr_dir outdir"
    parser = OptionParser(usage=usage)

    (options, args) = parser.parse_args()

    indir = args[0]
    outdir = args[1]

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    files = sorted(glob(os.path.join(indir, '*.jsonlist')))
    for infile in files:
        outlines = []
        print(infile)
        with open(infile) as f:
            lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            speaker = line['speaker']
            # normalize spacing
            speaker = re.sub(r'\s+', ' ', speaker)
            text = line['text']
            # remove speaker name from start of speech
            assert text[:len(speaker)] == speaker
            text = text[len(speaker)+1:].strip()
            text = normalize_to_stanford(text)
            line['text'] = text
            outlines.append(line)

        outfile = os.path.join(outdir, os.path.basename(infile))
        with open(outfile, 'w') as f:
            for line in outlines:
                f.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
