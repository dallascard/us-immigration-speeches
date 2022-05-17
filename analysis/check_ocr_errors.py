import os
import json
from optparse import OptionParser
from collections import defaultdict, Counter

from tqdm import tqdm


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--hein-dir', type=str, default='data/speeches/Congress/hein-bound_tokenized/',
                      help='Hein bound dir: default=%default')
    parser.add_option('--first', type=int, default=46,
                      help='First congress: default=%default')
    parser.add_option('--last', type=int, default=70,
                      help='Last congress: default=%default')
    parser.add_option('--targets', type=str, default='immigration,chinese,quota,citizen',
                      help='Comma-separated list of target terms: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()
    hein_dir = options.hein_dir
    first = options.first
    last = options.last
    targets = options.targets.split(',')

    token_counter = Counter()

    for congress in tqdm(range(first, last+1)):
        infile = os.path.join(hein_dir, 'speeches_' + str(congress).zfill(3) + '.jsonlist')
        with open(infile) as f:
            lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            for sent in line['tokens']:
                token_counter.update([t.lower() for t in sent])

    for target in targets:
        target_counter = Counter()
        for token, count in tqdm(token_counter.items()):
            if abs(len(token) - len(target)) <= 1:
                dist = levenshteinDistance(target, token)
                if dist == 1:
                    target_counter[token] = count

        print(target, token_counter[target])
        for term, count in target_counter.most_common(n=10):
            print(term, count)


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

if __name__ == '__main__':
    main()
