import os
import json
from optparse import OptionParser
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm

from speech_selection.common import match_tokens
from speech_selection.query_terms import early


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--hein-bound-dir', type=str, default='data/speeches/Congress/hein-bound_tokenized/',
                      help='Hein bound input directory: default=%default')
    parser.add_option('--outfile', type=str, default='data/speeches/Congress/keyword_segments_new/keyword_segments_43-73.jsonlist',
                      help='Output directory: default=%default')
    parser.add_option('--use-sents', action="store_true", default=False,
                      help='Use sentences rather than tokens (avoid excess spaces): default=%default')

    (options, args) = parser.parse_args()

    hein_bound_dir = options.hein_bound_dir
    outfile = options.outfile
    use_sents = options.use_sents

    outdir = os.path.split(outfile)[0]
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    first = 43
    last_bound = 73
    files = []
    for congress in range(first, last_bound+1):
        files.append(os.path.join(hein_bound_dir, 'speeches_' + str(congress).zfill(3) + '.jsonlist'))
    files.sort()
    print(len(files))

    chunk_lengths = Counter()

    count = 0
    total_count = 0

    outlines = []
    for infile in files:
        basename = infile.split('/')[-1].split('.')[0]
        with open(infile) as f:
            lines = f.readlines()
        lines = [json.loads(line) for line in lines]

        print(basename, len(lines))
        for line_i, line in enumerate(tqdm(lines)):
            tokenized_sents = line['tokens']
            if use_sents:
                sents = line['sents']
            else:
                sents = []
                for sent in tokenized_sents:
                    sents.append(' '.join(sent))
            assert len(sents) == len(tokenized_sents)
            line_id = line['id']

            n_sents = len(tokenized_sents)
            for sent_i, tokens in enumerate(tokenized_sents):
                match = match_tokens(tokens, early)

                if match:
                    chunk = ' '.join(sents[max(0, sent_i-3): min(n_sents, sent_i+4)])
                    outlines.append({'infile': basename, 'id': str(line_id) + '_' + str(sent_i), 'text': chunk})
                    chunk_lengths.update([len(chunk.split())])
                    count += 1
                    total_count += 1

    # write all congresses together
    with open(outfile, 'w') as f_o:
        for line in outlines:
            f_o.write(json.dumps(line) + '\n')

    print(total_count)


if __name__ == '__main__':
    main()
