import os
import re
import json
from glob import glob
from collections import Counter, defaultdict
from optparse import OptionParser

from tqdm import tqdm

from analysis.metaphor_terms import get_metaphor_terms


# Script to identify mentions of metaphorical terms for the purpose of validation

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--infile', type=str, default='data/speeches/Congress/imm_speeches_parsed.jsonlist',
                      help='Input file from export.export_imm_speeches_parsed.py: default=%default')
    parser.add_option('--outdir', type=str, default='data/speeches/Congress/metaphors/',
                      help='Output file: default=%default')
    #parser.add_option('--countfile', type=str, default='/u/scr/nlp/data/congress/immigration/predictions/imm_mention_sents_parsed_counts.json',
    #                  help='Output count file: default=%default')
    parser.add_option('--min-length', type=int, default=4,
                      help='Min length in tokens (per sentence): default=%default')

    (options, args) = parser.parse_args()

    infile = options.infile
    min_length = options.min_length
    outdir = options.outdir

    metaphor_terms = get_metaphor_terms()
    metaphor_term_sets = {}
    for metaphor, terms in metaphor_terms.items():
        metaphor_term_sets[metaphor] = set(terms)

    term_counter_by_metaphor = defaultdict(Counter)

    print("Loading data")
    with open(infile) as f:
        lines = f.readlines()

    outlines_by_metaphor = defaultdict(list)

    for line in tqdm(lines):
        line = json.loads(line)
        speech_id = line['id']
        tokens = line['tokens']
        lemmas = line['lemmas']
        tags = line['tags']

        for sent_i, orig_tokens in enumerate(tokens):
            for metaphor, target_terms in metaphor_term_sets.items():
                # convert to lower case and do an initial filter
                lower_tokens = [t.lower() for t in orig_tokens]
                if len(lower_tokens) >= min_length and len(set(lower_tokens).intersection(target_terms)) > 0:

                    # Save the sentence with metadata
                    outlines_by_metaphor[metaphor].append({'id': speech_id, 'sent_index': sent_i, 'tokens': orig_tokens, 'lemmas': lemmas[sent_i], 'tags': tags[sent_i]})
                    # Keep track of which terms are found most frequently
                    term_counter_by_metaphor[metaphor].update(set(lower_tokens).intersection(target_terms))

    for metaphor in metaphor_term_sets:
        print(metaphor, len(outlines_by_metaphor[metaphor]))
        outfile = os.path.join(outdir, 'metaphor_mention_sents_parsed_{:s}.jsonlist'.format(metaphor))
        with open(outfile, 'w') as f:
            for line in outlines_by_metaphor[metaphor]:
                f.write(json.dumps(line) + '\n')

        outfile = os.path.join(outdir, 'metaphor_mention_sents_parsed_counts_{:s}.jsonlist'.format(metaphor))
        with open(outfile, 'w') as f:
            json.dump(term_counter_by_metaphor[metaphor].most_common(), f, indent=2, sort_keys=False)


if __name__ == '__main__':
    main()
