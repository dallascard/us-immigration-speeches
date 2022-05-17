import os
from glob import glob
from optparse import OptionParser

import json
import tqdm

import spacy


def main():
    usage = "%prog hein-bound-dir output-dir"
    parser = OptionParser(usage=usage)
    parser.add_option('--first', type=int, default=43,
                      help='First congress: default=%default')
    parser.add_option('--last', type=int, default=111,
                      help='Last congress: default=%default')

    (options, args) = parser.parse_args()

    indir = args[0]
    outdir = args[1]
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    first = options.first
    last = options.last

    print("Loading spacy")
    nlp = spacy.load("en_core_web_sm")

    for congress in range(first, last+1):
        infile = os.path.join(indir, 'speeches_' + str(congress).zfill(3) + '.txt')
        descr_file = os.path.join(indir, 'descr_' + str(congress).zfill(3) + '.txt')
        print(infile)
        basename = os.path.splitext(os.path.basename(infile))[0]
        outlines = []
        speech_dates = {}
        with open(descr_file, encoding='Windows-1252') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.split('|')
            speech_id = parts[0]
            date = parts[2]
            speech_dates[speech_id] = date

        with open(infile, encoding='Windows-1252') as f:
            lines = f.readlines()
        for line in tqdm.tqdm(lines):
            line = line.strip()
            parts = line.split('|')
            line_id = parts[0]
            # drop the header
            if line_id != 'speech_id':
                date = speech_dates[line_id]
                # skip one day that has is corrupted by data from 1994
                if date != '18940614':
                    text = ' '.join(parts[1:])
                    parsed = nlp(text)
                    sents = []
                    tokens = []
                    for sent in parsed.sents:
                        sents.append(sent.text)
                        tokens.append([token.text for token in sent])

                    assert len(sents) == len(tokens)

                    rejoined_sents = []
                    rejoined_tokens = []
                    if len(sents) > 0:
                        current_sent = sents[0]
                        current_tokens = tokens[0]
                        if len(sents) > 1:
                            for sent_i in range(1, len(sents)):
                                # look to see if this might be a false sentence break
                                if sents[sent_i-1][-1] == '.' and (sents[sent_i][0].islower() or sents[sent_i][0].isdigit() or sents[sent_i][0] == '$' or sents[sent_i][0] == '%'):
                                    # if so, extend the previous sentence / tokens
                                    current_sent += ' ' + sents[sent_i]
                                    current_tokens.extend(tokens[sent_i])
                                else:
                                    # otherwise, add the previous to the list, and start a new one
                                    rejoined_sents.append(current_sent)
                                    rejoined_tokens.append(current_tokens)
                                    current_sent = sents[sent_i]
                                    current_tokens = tokens[sent_i]
                        # add the current to the list
                        rejoined_sents.append(current_sent)
                        rejoined_tokens.append(current_tokens)

                    outlines.append({'infile': basename, 'id': line_id, 'sents': rejoined_sents, 'tokens': rejoined_tokens})

        outfile = os.path.join(outdir, basename + '.jsonlist')
        print("Saving {:d} lines to {:s}".format(len(outlines), outfile))
        with open(outfile, 'w') as fo:
            for line in outlines:
                fo.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
