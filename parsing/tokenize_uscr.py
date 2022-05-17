import os
import re
from optparse import OptionParser

import json
import tqdm

import spacy


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--indir', type=str, default='data/speeches/Congress/uscr_processed/',
                      help='Processed USCR dir: default=%default')
    parser.add_option('--outdir', type=str, default='data/speeches/Congress/uscr_tokenized/',
                      help='Output dir: default=%default')
    parser.add_option('--first', type=int, default=104,
                      help='First congress [104-116]: default=%default')
    parser.add_option('--last', type=int, default=116,
                      help='Last congress [104-116]: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    indir = options.indir
    outdir = options.outdir
    first = options.first
    last = options.last

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("Loading spacy")
    nlp = spacy.load("en_core_web_sm")

    for n in range(first, last+1):
        print(n)
        infile = os.path.join(indir, 'speeches_' + str(n) + '.jsonlist')
        basename = os.path.splitext(os.path.basename(infile))[0]
        outlines = []
        with open(infile) as f:
            lines = f.readlines()

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

            assert len(sents) == len(tokens)

            outlines.append({'infile': basename, 'id': line_id, 'sents': sents, 'tokens': tokens})

        outfile = os.path.join(outdir, basename + '.jsonlist')
        print("Saving {:d} lines to {:s}".format(len(outlines), outfile))
        with open(outfile, 'w') as fo:
            for line in outlines:
                fo.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
