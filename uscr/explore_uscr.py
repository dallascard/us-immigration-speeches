import os
import json
from glob import glob
from optparse import OptionParser
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm


def main():
    usage = "%prog cr_raw_dir"
    parser = OptionParser(usage=usage)

    (options, args) = parser.parse_args()

    basedir = args[0]

    for year in range(2019, 2020):
        speaker_counter = Counter()
        files = sorted(glob(os.path.join(basedir, str(year), '*', 'json', '*.json')))
        indices = np.arange(len(files))
        n_to_sample = int(len(indices) * 0.1)
        subset = np.random.choice(indices, n_to_sample, replace=False)
        for i in tqdm(subset):
            with open(files[i]) as f:
                data = json.load(f)
                content = data['content']
                for entry in content:
                    speaker_counter[entry['speaker']] += 1
                    if entry['turn'] > 0:
                        print(files[i], entry)
        print(year)
        for k, c in speaker_counter.most_common(n=30):
            print(k, c)


if __name__ == '__main__':
    main()
