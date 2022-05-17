import os
import re
import json
from optparse import OptionParser

import spacy
from tqdm import tqdm


# RUN Dependency paring on the USCR congressional speeches
# This is mostly done for looking for good MWEs and collocations
# so, it will re-attach each possessive 's, but will not fix bad sentence segmentation


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--indir', type=str, default='data/speeches/Congress/uscr_processed/',
                      help='Hein bound dir: default=%default')
    parser.add_option('--outdir', type=str, default='data/speeches/Congress/uscr_parsed/',
                      help='Hein bound dir: default=%default')
    parser.add_option('--first', type=int, default=104,
                      help='First congress: default=%default')
    parser.add_option('--last', type=int, default=116,
                      help='Last congress: default=%default')
    #parser.add_option('--encoding', type=str, default='Windows-1252',
    #                  help='Infile encoding: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    indir = options.indir
    outdir = options.outdir
    first = options.first
    last = options.last
    #encoding = options.encoding

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("Loading spacy")
    nlp = spacy.load("en_core_web_sm")

    for congress in range(first, last+1):
        infile = os.path.join(indir, 'speeches_' + str(congress).zfill(3) + '.jsonlist')

        with open(infile) as f:
            lines = f.readlines()

        outlines = []
        for line in tqdm(lines):
            line = json.loads(line)
            line_id = line['id']
            # drop the header
            if line_id != 'speech_id':
                text = line['text']

                # parse the text
                parsed = nlp(text)

                # collect features to be saved
                sents = []
                tokens = []
                lemmas = []
                tags = []
                deps = []
                heads = []
                for sent in parsed.sents:
                    text = re.sub(r',', '.', sent.text)
                    sents.append(text)
                    tokens.append([re.sub(r',', '.', token.text) for token in sent])
                    lemmas.append([re.sub(r',', '.', token.lemma_) for token in sent])
                    tags.append([token.tag_ for token in sent])
                    deps.append([token.dep_ for token in sent])
                    heads.append([token.head.i - sent.start for token in sent])

                outlines.append({'id': line_id, 'tokens': tokens, 'lemmas': lemmas, 'tags': tags, 'deps': deps, 'heads': heads})

        outfile = os.path.join(outdir, 'speeches_' + str(congress).zfill(3) + '.jsonlist')
        print("Saving {:d} lines to {:s}".format(len(outlines), outfile))
        with open(outfile, 'w') as fo:
            for line in outlines:
                fo.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
