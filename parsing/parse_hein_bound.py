import os
import json
from optparse import OptionParser

import spacy
from tqdm import tqdm


# RUN Dependency paring on the original congressional speeches
# This is mostly done for looking for good MWEs and collocations
# so, it will re-attach each possessive 's, but will not fix bad sentence segmentation


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--indir', type=str, default='data/speeches/Congress/hein-bound/',
                      help='Hein bound dir: default=%default')
    parser.add_option('--outdir', type=str, default='data/speeches/Congress/hein-bound_parsed/',
                      help='Hein bound dir: default=%default')
    parser.add_option('--first', type=int, default=43,
                      help='First congress: default=%default')
    parser.add_option('--last', type=int, default=111,
                      help='Last congress: default=%default')
    parser.add_option('--encoding', type=str, default='Windows-1252',
                      help='Infile encoding: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    indir = options.indir
    outdir = options.outdir
    first = options.first
    last = options.last
    encoding = options.encoding

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("Loading spacy")
    nlp = spacy.load("en_core_web_sm")

    for congress in range(first, last+1):
        infile = os.path.join(indir, 'speeches_' + str(congress).zfill(3) + '.txt')
        descr_file = os.path.join(indir, 'descr_' + str(congress).zfill(3) + '.txt')
        basename = os.path.splitext(os.path.basename(infile))[0]

        # first read the description file to get speech dates
        speech_dates = {}
        with open(descr_file, encoding=encoding) as f:
            lines = f.readlines()
        for line in lines:
            parts = line.split('|')
            speech_id = parts[0]
            date = parts[2]
            speech_dates[speech_id] = date

        with open(infile, encoding='Windows-1252') as f:
            lines = f.readlines()

        outlines = []
        for line in tqdm(lines):
            line = line.strip()
            parts = line.split('|')
            line_id = parts[0]
            # drop the header
            if line_id != 'speech_id':
                date = speech_dates[line_id]
                # skip one day that has is corrupted by data from 1994
                if date != '18940614':
                    text = ' '.join(parts[1:])

                    # parse the text
                    parsed = nlp(text)

                    # reattach possessive 's's
                    possessives = [token.i for token in parsed if token.tag_ == 'POS' and token.text == "'s" and token.sent.start != token.i]
                    with parsed.retokenize() as retokenizer:
                        for pos in possessives:
                            retokenizer.merge(parsed[pos-1:pos+1])

                    # collect features to be saved
                    sents = []
                    tokens = []
                    lemmas = []
                    tags = []
                    deps = []
                    heads = []
                    for sent in parsed.sents:
                        sents.append(sent.text)
                        tokens.append([token.text for token in sent])
                        lemmas.append([token.lemma_ for token in sent])
                        tags.append([token.tag_ for token in sent])
                        deps.append([token.dep_ for token in sent])
                        heads.append([token.head.i - sent.start for token in sent])

                    outlines.append({'id': line_id, 'tokens': tokens, 'lemmas': lemmas, 'tags': tags, 'deps': deps, 'heads': heads})

        outfile = os.path.join(outdir, 'speeches_' + str(congress).zfill(3) + '.txt')
        print("Saving {:d} lines to {:s}".format(len(outlines), outfile))
        with open(outfile, 'w') as fo:
            for line in outlines:
                fo.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
