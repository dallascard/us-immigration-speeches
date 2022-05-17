import os
import json
from optparse import OptionParser
from collections import defaultdict, Counter

from tqdm import tqdm

from speech_selection.common import match_tokens
from speech_selection.query_terms import early, mid, modern


# Select paragraphs from Presidential speeches using keywords

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--infile', type=str, default='data/speeches/Presidential/paragraphs.tokenized.jsonlist',
                      help='Infile: default=%default')
    parser.add_option('--outfile', type=str, default='data/speeches/Presidential/paragraphs.keywords.jsonlist',
                      help='Outfile: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    infile = options.infile
    outfile = options.outfile

    # get years dividing keywords
    early_last = 1934
    modern_first = 1957

    # read in the tokenized paragraphs
    with open(infile) as f:
        lines = f.readlines()
    lines = [json.loads(line) for line in lines]

    outlines = []
    # process each paragraph
    for line_i, line in enumerate(tqdm(lines)):
        tokenized_sents = line['tokens']
        year = line['year']

        any_match = False
        # check each sentence in paragraph
        for sent_i, tokens in enumerate(tokenized_sents):
            # match to a set of keywords depending on year
            if year <= early_last:
                match = match_tokens(tokens, early)
            elif year < modern_first:
                match = match_tokens(tokens, mid)
            else:
                match = match_tokens(tokens, modern)

            if match:
                any_match = True

        # keep the paragraph if any of the sentences match
        if any_match:
            outlines.append(line)

    # write all congresses together
    print("Saving {:d} paragraphs with keywords".format(len(outlines)))
    with open(outfile, 'w') as f_o:
        for line in outlines:
            f_o.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
