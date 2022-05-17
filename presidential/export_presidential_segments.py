import os
import json
from optparse import OptionParser
from collections import defaultdict, Counter

from tqdm import tqdm

from speech_selection.common import match_tokens
from speech_selection.query_terms import early, mid, modern
from parsing.common import normalize_to_stanford


# Converts Presidential documents to single paragraphs:
# Takes as input, the output of scrapers/app/combine_categories.py

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--orig-app-file', type=str, default='/<path_to_scrapers>/data/app/all.jsonlist',
                      help='Original file with all documents from scrapers/app: default=%default')
    parser.add_option('--outdir', type=str, default='data/speeches/Presidential/',
                      help='Output directory: default=%default')

    (options, args) = parser.parse_args()

    outdir = options.outdir
    orig_file = options.orig_app_file
    outfile = os.path.join(outdir, 'paragraphs.jsonlist')
    counts_file = os.path.join(outdir, 'paragraph_counts.json')
    doc_counts_file = os.path.join(outdir, 'doc_counts.json')

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("Reading data")
    with open(orig_file) as f:
        lines = f.readlines()
    print("Loaded {:d} documents".format(len(lines)))

    # excluded if followed by [NAME] then a "." or ":"
    unigram_titles = {
        'mr.',
        'mrs.',
        'ms.',
        'dr.',
        'sen.',
        'colonel',
        'admiral',
        'director',
        'secretary',
        'chairman',
        'general',
        'ambassador',
        'administrator',
        'premier',
        'chancellor',
        'governor',
    }

    bigram_titles = {'prime minister',
                     'vice president',
                     'vice premier',
                     'attorney general',
                     'major general',
                     }

    # exclude if followed by "." or ":"
    paragraph_starts_to_exclude = ['q',
                                   'moderator',
                                   'senior administration official',
                                   'the vice president',
                                   'the prime minister',
                                   'the premier',
                                   'the vice premier',
                                   'the chancellor',
                                   'the first lady',
                                   'audience',
                                   'crowd',
                                   'reporter',
                                   'the press',
                                   'child',
                                   'resident',
                                   'president mubarak',
                                   'president yeltsin',
                                   'president putin',
                                   'president karzai',
                                   'president mitterrand',
                                   'president macron',
                                   'president gorbachev'
                                   ]

    # include if followed by "." or ":"
    presidential_starts = ['the president',
                           'president bush',
                           'president trump',
                           'president obama',
                           'president clinton',
                           'president reagan',
                           'president carter'
                           ]

    outlines = []
    n_excluded = 0

    paragraphs_per_year_by_president = defaultdict(Counter)
    documents_per_year_by_president = defaultdict(Counter)

    for line_i, line in enumerate(tqdm(lines)):
        line = json.loads(line)
        url = line['url']
        paragraphs = line['text']
        date = line['date']
        parts = date.strip().split()
        year = int(parts[-1])
        person = line['person']
        documents_per_year_by_president[person][year] += 1

        # Look for lines that indicate multiple speakers
        # and try to identify them as the start of lines by the president or non president speakers
        line_types = []
        for paragraph in paragraphs:
            president = False
            non_president = False
            text = paragraph.lower()
            for s in presidential_starts:
                if len(text) > len(s):
                    if text.startswith(s) and (text[len(s)] == '.' or text[len(s)] == ':'):
                        president = True
                        break
            if not president:
                for s in paragraph_starts_to_exclude:
                    if len(text) > len(s):
                        if text.startswith(s) and (text[len(s)] == '.' or text[len(s)] == ':'):
                            non_president = True
                            break
            if not president and not non_president:
                tokens = text.split()
                if len(tokens) >= 3:
                    if tokens[0] in unigram_titles and (tokens[1][-1] == '.' or tokens[1][-1] == ':'):
                        non_president = True
                    elif tokens[0] + ' ' + tokens[1] in bigram_titles and (tokens[2][-1] == '.' or tokens[2][-1] == ':'):
                        non_president = True
            if president:
                line_type = 1
            elif non_president:
                line_type = 2
            else:
                line_type = 0
            line_types.append(line_type)

        if sum(line_types) == 0:
            # no indication of multiple speakers; just keep everything
            for p_i, p in enumerate(paragraphs):
                text = normalize_to_stanford(p)
                outlines.append({'id': url + '_' + str(p_i).zfill(4), 'text': text, 'year': year, 'prefix': ''})
                paragraphs_per_year_by_president[person][year] += 1
        else:
            # for multiple speakers, assume it starts with the president
            president = True
            for p_i, p in enumerate(paragraphs):
                # use these lines to switch between president and non-president
                if line_types[p_i] == 1:
                    president = True
                elif line_types[p_i] == 2:
                    president = False
                # only keep lines from the president
                if president:
                    # remove "The President." or equivalent from the start, if present
                    text = None
                    prefix = ''
                    text_lower = p.lower()
                    for s in presidential_starts:
                        if len(p) > len(s):
                            if text_lower.startswith(s) and (p[len(s)] == '.' or p[len(s)] == ':'):
                                prefix = p[:len(s)+1]
                                text = p[len(s)+1:].strip()
                                break
                    if text is None:
                        text = p
                    text = normalize_to_stanford(text)
                    outlines.append({'id': url + '_' + str(p_i).zfill(4), 'text': text, 'year': year, 'prefix': prefix})
                    paragraphs_per_year_by_president[person][year] += 1
                else:
                    n_excluded += 1

    print(len(outlines), "included")
    print(n_excluded, "excluded")

    with open(outfile, 'w') as f:
        for line in outlines:
            f.write(json.dumps(line) + '\n')

    with open(counts_file, 'w') as f:
        json.dump(paragraphs_per_year_by_president, f, indent=2)

    with open(doc_counts_file, 'w') as f:
        json.dump(documents_per_year_by_president, f, indent=2)


if __name__ == '__main__':
    main()
