import os
import json
from optparse import OptionParser
from collections import defaultdict, Counter

import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer

# Combine the keyword and non-keyword segmets into one file / dataframe


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--pred-dir', type=str, default='data/speeches/Congress/',
                      help='Predictions dir: default=%default')
    parser.add_option('--first-congress', type=int, default=43,
                      help='First congress: default=%default')
    parser.add_option('--last-congress', type=int, default=116,
                      help='Last congress: default=%default')
    parser.add_option('--test-prop', type=float, default=0.05,
                      help='Proportion to use as test data: default=%default')
    parser.add_option('--seed', type=int, default=42,
                      help='Random seed: default=%default')
    parser.add_option('--lemmatize', action="store_true", default=False,
                      help='Lemmatize with NLTK: default=%default')


    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    pred_dir = options.pred_dir
    first_congress = options.first_congress
    last_congress = options.last_congress
    test_prop = options.test_prop
    seed = options.seed
    lemmatize = options.lemmatize

    if lemmatize:
        print("Loading spacy")
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
 

    print("Loading data")
    with open(os.path.join(pred_dir, 'imm_segments_with_tone_and_metadata.jsonlist')) as f:
        lines = f.readlines()

    labels = ['anti', 'neutral', 'pro']

    print("Getting segments")
    outlines = []
    for line in tqdm(lines):
        line = json.loads(line)
        congress = int(line['congress'])
        if first_congress <= congress <= last_congress:
            segment_id = line['speech_id'] + '_' + str(line['segment'])
            text = line['text']
            tokens = text.split()
            if lemmatize:
                lemmatized = nlp(text)
                tokens = [token.lemma_ for token in lemmatized]
                #tokens = [lemmatizer.lemmatize(t) for t in tokens]
            probs = [line['anti'], line['neutral'], line['pro']]
            label = labels[np.argmax(probs)]
            outlines.append({'id': segment_id, 'congress': congress, 'date': line['date'], 'tokens': [tokens], 'label': label})

    print("Splitting")
    np.random.seed(seed)
    np.random.shuffle(outlines)
    n_test = int(len(outlines) * test_prop)

    print(n_test, len(outlines)-n_test)

    print("Saving")
    if lemmatize:
        outfile = os.path.join(pred_dir, 'imm_segments_with_tone_labels_{:d}-{:d}_lemmas_test.jsonlist').format(first_congress, last_congress)        
    else:
        outfile = os.path.join(pred_dir, 'imm_segments_with_tone_labels_{:d}-{:d}_test.jsonlist').format(first_congress, last_congress)
    with open(outfile, 'w') as f:
        for line in outlines[:n_test]:
            f.write(json.dumps(line) + '\n')

    if lemmatize:
        outfile = os.path.join(pred_dir, 'imm_segments_with_tone_labels_{:d}-{:d}_lemmas_train.jsonlist').format(first_congress, last_congress)    
    else:
        outfile = os.path.join(pred_dir, 'imm_segments_with_tone_labels_{:d}-{:d}_train.jsonlist').format(first_congress, last_congress)    
    with open(outfile, 'w') as f:
        for line in outlines[n_test:]:
            f.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
