import os
import json
from optparse import OptionParser
from collections import defaultdict, Counter
from xml.dom import NoModificationAllowedErr

import numpy as np
from transformers import BertTokenizer

from analysis.identify_immigrant_mentions import basic_terms, post_group_terms, group_terms
from analysis.group_terms import countries, nationalities, early_chinese_terms, european_countries, modern_mexican_terms
from analysis.metaphor_terms import metaphor_terms

# Get a set of random nouns in the bert vocabulary

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='data/speeches/Congress',
                      help='Base directory: default=%default')
    parser.add_option('--min-count', type=int, default=500,
                      help='Min count for inclusion: default=%default')
    parser.add_option('--seed', type=int, default=42,
                      help='Random seed: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    min_count = options.min_count
    seed = options.seed
    np.random.seed(seed)

    counts_dir = os.path.join(basedir, 'basic_counts')

    with open(os.path.join(counts_dir, 'noun_counts.json')) as f:
        data = json.load(f)

    tokenizer_class = BertTokenizer

    tokenizer = tokenizer_class.from_pretrained('bert-base-uncased')

    vocab = tokenizer.vocab

    # exclude all existing metaphor, immigration, and nationality terms from random selection
    exclusions = set()
    for metaphor, terms in metaphor_terms.items():
        if metaphor != 'random':
            exclusions.update(terms)
    exclusions.update(basic_terms)
    exclusions.update(post_group_terms)
    for group, terms in group_terms.items():
        exclusions.update(terms)
    exclusions.update(early_chinese_terms)
    exclusions.update([t.lower() for t in european_countries])
    exclusions.update(modern_mexican_terms)
    for country, terms in nationalities.items():
        exclusions.update([t.lower() for t in terms])
    for country, terms in countries.items():
        exclusions.update([t.lower() for t in terms])
    print(exclusions)

    print(len(exclusions))
    valid_counts = {noun: count for noun, count in data.items() if noun in vocab and count >= min_count and noun not in exclusions} 
    valid_nouns = sorted(valid_counts)
    print(len(valid_nouns))    
 
    subset = np.random.choice(valid_nouns, size=50, replace=False)

    counts = [data[noun] for noun in subset]
    order = np.argsort(counts)
    for i in order:
        print(subset[i], counts[i])

    with open(os.path.join(counts_dir, 'random_nouns.txt'), 'w') as f:
        for noun in subset:
            f.write(noun + '\n')

   

if __name__ == '__main__':
    main()
