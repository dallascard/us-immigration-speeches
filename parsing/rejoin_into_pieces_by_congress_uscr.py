import os
import json
from optparse import OptionParser

import numpy as np

# Same as rejoin_into_pieces, except output one file per congress

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--uscr-dir', type=str, default='data/speeches/Congress/uscr_tokenized',
                      help='Issue: default=%default')
    parser.add_option('--outdir', type=str, default='data/speeches/Congress/uscr_segments',
                      help='Issue: default=%default')
    parser.add_option('--first', type=int, default=104,
                      help='First congress: default=%default')
    parser.add_option('--last', type=int, default=116,
                      help='Last congress: default=%default')
    parser.add_option('--max', type=int, default=375,
                      help='Max tokens per block: default=%default')
    parser.add_option('--keep-boundaries', action="store_true", default=False,
                      help='Output each sentence on a separate line: default=%default')
    #parser.add_option('--replace-periods', action="store_true", default=False,
    #                  help='Change periods that look wrong to commas: default=%default')
    parser.add_option('--use-sents', action="store_true", default=False,
                      help='Use sentences rather than tokens (avoid excess spaces): default=%default')

    (options, args) = parser.parse_args()

    uscr_dir = options.uscr_dir
    outdir = options.outdir
    first = options.first
    last = options.last
    max_length = options.max
    keep_boundaries = options.keep_boundaries
    #replace_periods = options.replace_periods
    use_sents = options.use_sents

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for congress in range(first, last+1):
        infile = os.path.join(uscr_dir, 'speeches_' + str(congress) + '.jsonlist')

        print(infile)
        with open(infile) as f:
            lines = f.readlines()

        lengths = []
        outlines = []
        # read each speech, one by one
        for line in lines:
            line = json.loads(line)
            speech_id = line['id']  # speehch id
            if use_sents:
                sents = line['sents']  # list of strings
            else:
                sents = line['tokens']  # list of (list of strings)

            joined_sents = []
            # convert input to list of strings
            for sent in sents:
                if use_sents:
                    joined_sents.append(sent)
                else:
                    joined_sents.append(' '.join(sent))

            # then connect these sentence into blocks of up to max_length tokens or keep as sentences:
            if keep_boundaries:
                for s_i, sent in enumerate(joined_sents):
                    lengths.append(len(sent))
                    outlines.append({'id': speech_id + '_s' + f'{s_i:03}', 'text': sent})
            else:
                cur_length = 0
                output_blocks = []
                for sent in joined_sents:
                    sent_len = len(sent.split())
                    if len(output_blocks) == 0:
                        output_blocks.append(sent)
                        cur_length += sent_len
                    elif cur_length + sent_len > max_length:
                        # append this as a new block and reset count
                        output_blocks.append(sent)
                        cur_length = sent_len
                    else:
                        output_blocks[-1] += ' ' + sent
                        cur_length += sent_len

                for block_i, block in enumerate(output_blocks):
                    block_length = len(block.split())
                    lengths.append(block_length)
                    outlines.append({'id': speech_id + '_b' + f'{block_i:03}', 'text': block})

        print(np.mean(lengths), np.median(lengths), np.max(lengths), sum([1 for length in lengths if length > max_length]) / len(lengths))
        longest = int(np.argmax(lengths))
        print(outlines[longest])

        outfile = os.path.join(outdir, 'segments-' + str(congress).zfill(3) + '.jsonlist')
        with open(outfile, 'w') as f:
            for line in outlines:
                f.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
