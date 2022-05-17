import os
import re
import json
from optparse import OptionParser
from collections import defaultdict, Counter

from tqdm import tqdm

from analysis.group_terms import get_subset_terms, get_nationalities, add_american


# Identify immigration mention sentences which mention various groups

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--imm-parsed-file', type=str, default='data/speeches/Congress/imm_mention_sents_parsed.jsonlist',
                      help='File with parsed sentences containing immigrant mentions (from analysis.identify_immigrant_mentions.py): default=%default')
    parser.add_option('--outdir', type=str, default='data/speeches/Congress/tagged_counts/',
                      help='Output dir: default=%default')

    (options, args) = parser.parse_args()
   
    imm_parsed_file = options.imm_parsed_file
    outdir = options.outdir

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("Loading parsed sentences with mentions of immigrants")
    with open(imm_parsed_file) as f:
        imm_sent_lines = f.readlines()
    imm_sent_lines = [json.loads(line) for line in imm_sent_lines]
    print(len(imm_sent_lines))

    imm_mention_sent_indices_by_group = defaultdict(list)

    # Get the terms for various groups
    nationalities = get_nationalities()
    american_terms, substitutions = add_american(nationalities)
    early_chinese_terms, european_countries, modern_mexican_terms = get_subset_terms()
    modern_hispanic_terms = get_modern_hispanic_terms()
    european_terms = set()
    # get the terms associated with European countries
    for country in european_countries:
        european_terms.add(country.lower())
        european_terms.update([term.lower() for term in nationalities[country]])
        european_terms.update([term.lower() for term in american_terms[country]])
    # convert the substitutions to lower case
    lower_substitutions = {k.lower(): s.lower() for k, s in substitutions.items()}    

    print(sorted(early_chinese_terms))
    print(sorted(modern_mexican_terms))
    print(sorted(modern_hispanic_terms))
    print(sorted(european_terms))
    print("Substitutions")
    for q in sorted(lower_substitutions):
        print(q, lower_substitutions[q])

    sub_counter = Counter()
    lines_by_group = Counter()
    for line in tqdm(imm_sent_lines):
        speech_id = line['id']
        sent_index = line['sent_index']
        sent_id = str(speech_id) + '_' + str(sent_index)

        tokens = [t.lower() for t in line['tokens']]

        # rejoin tokens into a string
        text = ' '.join(tokens)
        # do replacements (e.g., asian american -> asianamerican)
        for query, replacement in lower_substitutions.items():
            if query in text:
                sub_counter[query] += 1
                text = re.sub(query, replacement, text)
        # resplit into tokens
        tokens = set(text.split())
        # look for overlap with nationality terms
        if len(set(tokens).intersection(modern_mexican_terms)) > 0:
            lines_by_group['Mexican'] += 1
            imm_mention_sent_indices_by_group['Mexican'].append(sent_id)
        if len(set(tokens).intersection(modern_hispanic_terms)) > 0:
            lines_by_group['Hispanic'] += 1
            imm_mention_sent_indices_by_group['Hispanic'].append(sent_id)
        if len(set(tokens).intersection(european_terms)) > 0:
            lines_by_group['European'] += 1
            imm_mention_sent_indices_by_group['European'].append(sent_id)
        if len(set(tokens).intersection(early_chinese_terms)) > 0:
            lines_by_group['Chinese'] += 1
            imm_mention_sent_indices_by_group['Chinese'].append(sent_id)

    print("Substitutions:")
    for q, c in sub_counter.most_common(n=100):
        print(q, c)

    print("Mention sents per group")
    for g, c in lines_by_group.most_common():
        print(g, c)

    with open(os.path.join(outdir, 'imm_mention_sent_indices_by_group.json'), 'w') as f:
        json.dump(imm_mention_sent_indices_by_group, f, indent=2)


if __name__ == '__main__':
    main()

