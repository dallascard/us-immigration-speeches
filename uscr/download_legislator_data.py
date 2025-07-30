import os
import json
from glob import glob
from optparse import OptionParser
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm
import urllib.request 


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--outdir', type=str, default='data/congress/',
                      help='Output directory: default=%default')

    (options, args) = parser.parse_args()

    outdir = options.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    url = 'https://unitedstates.github.io/congress-legislators/legislators-historical.json'
    outfile = os.path.join(outdir, 'legislators-historical.json')    
    print("Downloading", url, "to", outfile)
    urllib.request.urlretrieve(url, outfile)

    with open(outfile) as f:
        legislators = json.load(f)
    print(len(legislators), 'historical legislators')

    print("Downloading", url, "to", outfile)
    url = 'https://unitedstates.github.io/congress-legislators/legislators-current.json'
    outfile = os.path.join(outdir, 'legislators-current.json')    
    urllib.request.urlretrieve(url, outfile)

    with open(outfile) as f:
        legislators_current = json.load(f)
    print(len(legislators_current), 'current legislators')

    for leg in legislators_current:
        legislators.append(leg)
    print(len(legislators), 'total legislators')

    print("Saving all to", os.path.join(outdir, 'legislators-all.json'))
    with open(os.path.join(outdir, 'legislators-all.json'), 'w') as f:
        json.dump(legislators, f, indent=2)


if __name__ == '__main__':
    main()
