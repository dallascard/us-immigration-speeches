import os
import json
from optparse import OptionParser
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


# Collect predictions from the short vs long classifier (which is really procedural vs not)

def main():
    usage = "%prog pred_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('--procedural-dir', type=str, default='data/speeches/Congress/procedural/',
                      help='Output directory (also dir with very short speech ids.json: default=%default')
    parser.add_option('--speech-dir', type=str, default='data/speeches/Congress/short_speeches/',
                      help='Directory with short speeches: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    pred_dir = args[0]

    data_dir = options.procedural_dir
    speech_dir = options.speech_dir

    procedural_ids = set()
    for congress in range(43, 117):
        infile = os.path.join(speech_dir, 'short_speeches_' + str(congress).zfill(3) + '.jsonlist')
        with open(infile) as f:
            lines = f.readlines()
        lines = [json.loads(line) for line in lines]
        speech_ids = [line['id'] for line in lines]

        infile = os.path.join(pred_dir, 'pred.probs.short_' + str(congress).zfill(3) + '.csv')
        df = pd.read_csv(infile, header=0, index_col=0)
        assert len(df) == len(lines)
        proc_probs = df['yes'].values

        for i, speech_id in enumerate(speech_ids):
            if proc_probs[i] > 0.5:
                procedural_ids.add(str(speech_id))

        print(congress, np.sum(proc_probs > 0.5) / len(proc_probs), len(procedural_ids))
    print(len(procedural_ids), "speeches classified as procedural")

    # add in very short speehces
    with open(os.path.join(data_dir, 'very_short_speech_ids.txt')) as f:
        very_short_speech_ids = f.readlines()
    very_short_speech_ids = [i.strip() for i in very_short_speech_ids]
    print(len(very_short_speech_ids), "very short speeches")

    all_ids = set(very_short_speech_ids).union(set(procedural_ids))
    all_ids = sorted(all_ids)
    print(len(all_ids), "total procedural speeches")

    with open(os.path.join(data_dir, 'procedural_speech_ids.txt'), 'w') as f:
        for i in all_ids:
            f.write(i + '\n')


if __name__ == '__main__':
    main()
