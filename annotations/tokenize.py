import os
import glob
import json
from optparse import OptionParser

import spacy
import pandas as pd
from tqdm import tqdm


# Consolidate the data per-segment, otehr than annotations, and to tokenization


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
    elif subset == 'early':
        first_round = 0
        last_round = 15
    elif subset == 'mid':
        first_round = 27
        last_round = 34
    else:
        raise ValueError("--subset must be early, mid, or modern")

    outdir = os.path.join('data', subset)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = os.path.join(outdir, 'texts.json')

    print("Loading spacy")
    nlp = spacy.load("en_core_web_sm")

    text_by_id = {}

    for r in range(first_round, last_round+1):
        if r < 10:
            files = glob.glob('data/round_0' + str(r) + '*/*.csv')
        else:
            files = glob.glob('data/round_' + str(r) + '*/*.csv')
        for f in files:
            df = pd.read_csv(f, header=0, index_col=0)
            item_ids = df['id'].values
            texts = df['text'].values
            congresses = df['congress'].values
            if 'p_immigration' not in df.columns:
                phase = 1
                df['p_immigration'] = 1.
            else:
                phase = 2
            sample_probs = df['p_immigration'].values
            for i, item_id in enumerate(item_ids):
                text_by_id[item_id] = {'text': texts[i], 'round': r, 'congress': congresses[i], 'sample_prob': sample_probs[i], 'phase': phase}
        print(r, len(text_by_id))

    lines = {}
    print("Processing documents")
    for item_id, data in tqdm(text_by_id.items()):
        text = data['text']
        parsed = nlp(text)
        tokens = []
        for sent in parsed.sents:
            sent_tokens = [token.text for token in sent]
            tokens.append(sent_tokens)
        lines[item_id] = {'text': text, 'tokens': tokens, 'round': int(data['round']), 'congress': int(data['congress']), 'sample_prob': float(data['sample_prob']), 'phase': int(data['phase'])}

    with open(outfile, 'w') as f:
        json.dump(lines, f, indent=2)


if __name__ == '__main__':
    main()
