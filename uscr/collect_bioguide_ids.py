import os
import json
from glob import glob
from optparse import OptionParser
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm


def main():
    usage = "%prog cr_raw_dir outfile.json legislators-all.json"
    parser = OptionParser(usage=usage)

    (options, args) = parser.parse_args()

    basedir = args[0]
    outfile = args[1]
    bioguide_file = args[2]

    with open(bioguide_file) as f:
        legislators = json.load(f)
    known_ids = set([person['id']['bioguide'] for person in legislators])

    bioguide_names = defaultdict(Counter)
    names_to_bioguide = defaultdict(Counter)

    for year in range(2012, 2021):
        speaker_counter = Counter()
        files = sorted(glob(os.path.join(basedir, str(year), '*', 'json', '*.json')))
        for infile in tqdm(files):
            with open(infile) as f:
                data = json.load(f)
                content = data['content']
                for entry in content:
                    try:
                        kind = entry['kind']
                        if kind == 'speech':
                            speaker = entry['speaker']
                            bioguide = entry['speaker_bioguide']
                            if bioguide is None or bioguide == 'None':
                                bioguide_names['None'][speaker] += 1
                            else:
                                bioguide_names[bioguide][speaker] += 1
                            names_to_bioguide[speaker][bioguide] += 1
                    except KeyError as e:
                        print(infile)
                        print(entry)
                        raise e

    bioguides = sorted(bioguide_names)
    lengths = [len(bioguide_names[bio]) for bio in bioguides]
    order = np.argsort(lengths)[::-1]

    print(bioguides[order[0]], bioguide_names[bioguides[order[0]]].most_common(n=10))

    for i in order[1:5]:
        print(bioguides[i], bioguide_names[bioguides[i]])

    names = sorted(names_to_bioguide)
    lengths = [len(names_to_bioguide[name]) for name in names]
    order = np.argsort(lengths)[::-1]
    for i in order[:5]:
        print(names[i], names_to_bioguide[names[i]])

    found = sum([1 if bio in known_ids else 0 for bio in bioguide_names])
    not_found = sum([0 if bio in known_ids else 1 for bio in bioguide_names])
    print(found, not_found)

    with open(outfile, 'w') as f:
        json.dump({bioguide: sorted(names) for bioguide, names in bioguide_names.items()}, f, indent=2)


if __name__ == '__main__':
    main()
