import os
import re
from optparse import OptionParser

import json
import tqdm

import spacy


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--infile', type=str, default='data/speeches/Presidential/paragraphs.jsonlist',
                      help='Processed USCR dir: default=%default')
    parser.add_option('--outfile', type=str, default='data/speeches/Presidential/paragraphs.tokenized.jsonlist',
                      help='Output dir: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    infile = options.infile
    outfile = options.outfile

    print("Loading spacy")
    nlp = spacy.load("en_core_web_sm")

    with open(infile) as f:
        lines = f.readlines()

    outlines = []
    for line in tqdm.tqdm(lines):
        line = json.loads(line)
        line_id = line['id']
        text = line['text']
        parsed = nlp(text)
        sents = []
        tokens = []
        for sent in parsed.sents:
            # convert commas to periods to match the Gentzkow data
            text = re.sub(r',', '.', sent.text)
            sents.append(text)
            tokens.append([re.sub(r',', '.', token.text) for token in sent])

        line['sents'] = sents
        line['tokens'] = tokens
        outlines.append(line)

    print("Saving {:d} lines to {:s}".format(len(outlines), outfile))
    with open(outfile, 'w') as fo:
        for line in outlines:
            fo.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
