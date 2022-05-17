import os
import glob
import json
from collections import Counter, defaultdict
from optparse import OptionParser

import numpy as np
import pandas as pd

from annotation_scripts.measure_agreement import levenshtein_distance


# Collect all annotations and export one per line (duplicating text)
# The output should be fed into the label aggregation repo


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    #parser.add_option('--issue', type=str, default='immigration',
    #                  help='Issue: default=%default')
    parser.add_option('--subset', type=str, default='early',
                      help='Subset to tokenize [early|mid|modern]: default=%default')

    (options, args) = parser.parse_args()

    subset = options.subset

    if subset == 'modern':
        first_round = 16
        last_round = 26
        n_example_lines = 0
    elif subset == 'early':
        first_round = 0
        last_round = 15
        n_example_lines = 8
    elif subset == 'mid':
        first_round = 27
        last_round = 34
        n_example_lines = 0
    else:
        raise ValueError("--subset must be early, mid, or modern")

    outdir = os.path.join('data', subset)

    renames = ['segment', 'congress', 'year', 'text', 'immigration', 'topic', 'tone', 'character', 'notes']

    immigration_counter = Counter()
    tone_counter = Counter()
    #topic_counter = Counter()
    character_counter = Counter()
    coder_counter = Counter()

    dfs = {}

    for r in range(first_round, last_round+1):

        dfs[r] = {}
        if r < 10:
            files = glob.glob('data/round_0' + str(r) + '*/*.tsv')
        else:
            files = glob.glob('data/round_' + str(r) + '*/*.tsv')
        print(r, len(files), files)
        for infile in files:
            basename = os.path.basename(infile)
            df = pd.read_csv(infile, header=0, index_col=None, sep='\t')
            columns = list(df.columns)
            columns[:len(renames)] = renames
            df.columns = columns
            if r == 0:
                # Drop lines that were provided as examples in the first round
                df = pd.DataFrame(df.values[n_example_lines:, :len(columns)], columns=columns)
            coder_name = basename.split()[0]
            coder_counter.update([coder_name])
            dfs[r][coder_name] = df
            immigration_counter.update(df['immigration'].values)
            tone_counter.update(df['tone'].values)
            #topic_counter.update(df['topic'].values)
            character_counter.update(df['character'].values)
        print(r, len(dfs[r]), sorted(dfs[r].keys()))
    print(coder_counter)

    # Deal with spelling mistakes in responses
    print("Mapping responses:")
    response_map = {}
    valid_responses = ['yes', 'no']
    null_val = 'nan'
    for response in immigration_counter:
        if type(response) == str:
            dists = []
            for possible_match in valid_responses:
                dist = levenshtein_distance(response, possible_match)
                dists.append(dist)
            closest_match = valid_responses[int(np.argmin(dists))]
            response_map[response] = closest_match
            print(response, '->', closest_match)

    print("\nMapping tone responses:")
    #tone_response_map = {}
    valid_responses = ['pro', 'neutral', 'anti']
    # set up some pre-specified mappings which exist in this data, and won't be found automatically
    tone_response_map = {'negative': 'anti',
                             'postive': 'pro',
                             'positive': 'pro'}
    for tone in tone_counter:
        if tone not in tone_response_map:
            if type(tone) == str:
                dists = []
                for possible_match in valid_responses:
                    dist = levenshtein_distance(tone, possible_match)
                    dists.append(dist)
                closest_match = valid_responses[int(np.argmin(dists))]
                tone_response_map[tone] = closest_match

    for tone in tone_counter:
        if type(tone) == str:
            print(tone, '->', tone_response_map[tone])

    immigration_lines = []
    tone_lines = []

    relevance_annotator_counter = Counter()
    tone_annotator_counter = Counter()

    relevant_descriptions = []

    for r in range(first_round, last_round+1):
        for coder_name, df in dfs[r].items():
            segments = df['segment'].values
            immigration = df['immigration'].values
            tone = df['tone'].values
            texts = df['text'].values
            topics = df['topic'].values
            worker = coder_name
            for i, segment in enumerate(segments):
                if type(topics[i]) == str:
                    topic = topics[i].strip()
                else:
                    topic = ''
                if type(immigration[i]) == str:
                    relevance_annotator_counter[coder_name] += 1
                    immigration_lines.append({'id': segment, 'round': r, 'text': texts[i], 'annotator': worker, 'label': response_map[immigration[i]], 'topic': topic})
                    if type(tone[i]) == str:
                        tone_annotator_counter[coder_name] += 1
                        tone_lines.append({'id': segment, 'round': r, 'text': texts[i], 'annotator': worker, 'label': tone_response_map[tone[i]]})

    print("{:d} relevance lines".format(len(immigration_lines)))
    print(relevance_annotator_counter.most_common())
    print("{:d} tone lines".format(len(tone_lines)))
    print(tone_annotator_counter.most_common())

    relevance_counter = Counter([line['label'] for line in immigration_lines])
    print(relevance_counter)

    tone_counter = Counter([line['label'] for line in tone_lines])
    print(tone_counter)

    with open(os.path.join(outdir, 'relevance_lines.jsonlist'), 'w') as f:
        for line in immigration_lines:
            f.write(json.dumps(line) + '\n')

    with open(os.path.join(outdir, 'tone_lines.jsonlist'), 'w') as f:
        for line in tone_lines:
            f.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
