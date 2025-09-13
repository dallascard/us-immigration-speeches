import os
import json
from glob import glob
from optparse import OptionParser

import tqdm
import spacy


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--hein-bound-dir', type=str, default='/data/stanford/hein-bound/',
                      help='Directory with the files from hein-bound.zip (from Stanford congress_text): default=%default')
    parser.add_option('--hein-daily-dir', type=str, default='/data/stanford/hein-daily/',
                      help='Directory with the files from hein-daily.zip (from Stanford congress_text): default=%default')
    parser.add_option('--outdir', type=str, default='data/hein-tokenized/',
                      help='Output directory for tokenized speeches: default=%default')
    parser.add_option('--first', type=int, default=43,
                      help='First congress: default=%default')
    parser.add_option('--last', type=int, default=114,
                      help='Last congress: default=%default')

    (options, args) = parser.parse_args()

    hein_bound_dir = options.hein_bound_dir
    hein_daily_dir = options.hein_daily_dir  
    outdir = options.outdir
    first = options.first
    last = options.last
    
    # Create the output directory if it does not exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("Loading spacy")
    nlp = spacy.load("en_core_web_sm")

    # Process the data for each session of Congress, one by one
    for congress in range(first, last+1):
        # Use the Hein bound data where possible
        if congress < 112:
            infile = os.path.join(hein_bound_dir, 'speeches_' + str(congress).zfill(3) + '.txt')
            descr_file = os.path.join(hein_bound_dir, 'descr_' + str(congress).zfill(3) + '.txt')
        else:
            infile = os.path.join(hein_daily_dir, 'speeches_' + str(congress).zfill(3) + '.txt')
            descr_file = os.path.join(hein_daily_dir, 'descr_' + str(congress).zfill(3) + '.txt')
        print(infile)
        basename = os.path.splitext(os.path.basename(infile))[0]
        outlines = []
        speech_dates = {}
        # Read in the description file
        with open(descr_file, encoding='Windows-1252') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.split('|')
            speech_id = parts[0]
            date = parts[2]
            speech_dates[speech_id] = date

        # Read in the speeches
        with open(infile, encoding='Windows-1252') as f:
            lines = f.readlines()

        # Process each speech one by one
        for line in tqdm.tqdm(lines):
            line = line.strip()
            # split into the speech id and speech text
            parts = line.split('|')
            line_id = parts[0]
            # drop the header
            if line_id != 'speech_id':
                # Get the date for this speech
                date = speech_dates[line_id]
                # skip one day that has is corrupted by data from 1994
                if date != '18940614':
                    # Join the everything to the right of the first |, in case there are |s in the speech
                    text = ' '.join(parts[1:])
                    # Parse the speech using spacy
                    parsed = nlp(text)
                    sents = []
                    tokens = []
                    # Collect sentences and tokens
                    for sent in parsed.sents:
                        sents.append(sent.text)
                        tokens.append([token.text for token in sent])

                    # Hueristically rejoin sentences that appear to be improperly split
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
                        
                        # add the rejoined data to the list
                        rejoined_sents.append(current_sent)
                        rejoined_tokens.append(current_tokens)

                    # Create a complete record for this speech
                    outlines.append({'infile': basename, 'id': line_id, 'sents': rejoined_sents, 'tokens': rejoined_tokens})

        # Save the parsed speeches for this session as a .jsonlist file
        outfile = os.path.join(outdir, basename + '.jsonlist')
        print("Saving {:d} lines to {:s}".format(len(outlines), outfile))
        with open(outfile, 'w') as fo:
            for line in outlines:
                fo.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
