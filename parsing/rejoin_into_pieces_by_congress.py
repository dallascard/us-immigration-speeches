import os
import json
from optparse import OptionParser

import numpy as np

# Same as rejoin_into_pieces, except output one file per congress

def main():
    usage = "%prog outdir"
    parser = OptionParser(usage=usage)
    parser.add_option('--hein-bound-dir', type=str, default='data/speeches/Congress/hein-bound-tokenized',
                      help='Issue: default=%default')
    parser.add_option('--hein-daily-dir', type=str, default='data/speeches/Congress/hein-daily-tokenized',
                      help='Issue: default=%default')
    parser.add_option('--first', type=int, default=43,
                      help='First congress: default=%default')
    parser.add_option('--last', type=int, default=114,
                      help='Last congress: default=%default')
    parser.add_option('--max', type=int, default=375,
                      help='Max tokens per block: default=%default')
    parser.add_option('--keep-boundaries', action="store_true", default=False,
                      help='Output each sentence on a separate line: default=%default')
    parser.add_option('--replace-periods', action="store_true", default=False,
                      help='Change periods that look wrong to commas: default=%default')
    parser.add_option('--use-sents', action="store_true", default=False,
                      help='Use sentences rather than tokens (avoid excess spaces): default=%default')

    (options, args) = parser.parse_args()

    outdir = args[0]

    hein_bound_dir = options.hein_bound_dir
    hein_daily_dir = options.hein_daily_dir
    first = options.first
    last = options.last
    max_length = options.max
    keep_boundaries = options.keep_boundaries
    replace_periods = options.replace_periods
    use_sents = options.use_sents

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for congress in range(first, last+1):
        if congress < 100:
            infile = os.path.join(hein_bound_dir, 'speeches_0' + str(congress) + '.jsonlist')
        elif congress > 111:
            infile = os.path.join(hein_daily_dir, 'speeches_' + str(congress) + '.jsonlist')
        else:
            infile = os.path.join(hein_bound_dir, 'speeches_' + str(congress) + '.jsonlist')

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
            n_tokens = 0
            n_chars = 0
            # first try to rejoin bad splits
            for tokens in sents:
                if use_sents:
                    # put the string in a list to conform to below
                    n_chars += len(tokens)
                    n_tokens += len(tokens.split())
                    tokens = [tokens]
                else:
                    n_chars += sum([len(t) for t in tokens]) + len(tokens) - 1
                    n_tokens += len(tokens)
                # if this is the first sentence, start a new string
                if len(joined_sents) == 0:
                    joined_sents.append(' '.join(tokens))
                # if it looks like this is not the start of a sentence:
                elif tokens[0][0].islower() or tokens[0][0].isdigit() or tokens[0][0] == '$' or tokens[0][0] == '%':
                    # change a period to a comma at the end of the last group, if it is there
                    if replace_periods and joined_sents[-1][-1] == '.':
                        joined_sents[-1] = joined_sents[-1][:-1] + ', ' + ' '.join(tokens)
                    else:
                        joined_sents[-1] += ' ' + ' '.join(tokens)
                    n_chars += 1
                # otherwise, start a new string
                else:
                    joined_sents.append(' '.join(tokens))

            if keep_boundaries:
                check = 0
                for s_i, sent in enumerate(joined_sents):
                    lengths.append(len(sent))
                    check += len(sent)
                    outlines.append({'id': speech_id + '_s' + f'{s_i:03}', 'text': sent})

                assert check == n_chars

            else:
                # then connect these sentence into blocks of up to max_length tokens or keep as sentences:
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

                check = 0
                for block_i, block in enumerate(output_blocks):
                    block_length = len(block.split())
                    check += block_length
                    lengths.append(block_length)
                    outlines.append({'id': speech_id + '_b' + f'{block_i:03}', 'text': block})

                try:
                    assert check == n_tokens
                except AssertionError as e:
                    print(n_tokens, check, len(outlines))

        print(np.mean(lengths), np.median(lengths), np.max(lengths), sum([1 for length in lengths if length > max_length]) / len(lengths))
        longest = int(np.argmax(lengths))
        print(outlines[longest])

        outfile = os.path.join(outdir, 'segments-' + str(congress).zfill(3) + '.jsonlist')
        with open(outfile, 'w') as f:
            for line in outlines:
                f.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
