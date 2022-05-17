import os
import json
from optparse import OptionParser
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
from time_periods.common import get_early_congress_range, get_modern_congress_range, year_to_congress


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--orig-app-file', type=str, default='/<path_to_scrapers>/data/app/all.jsonlist',
                      help='Original file with all documents from scrapers/app: default=%default')
    parser.add_option('--pres-keywords-file', type=str, default='data/speeches/Presidential/paragraphs.keywords.jsonlist',
                      help='Keyword paragraphs file: default=%default')
    parser.add_option('--pred-rel-early', type=str, default='data/speeches/Presidential/predictions/relevance_early/pred.pres.keywords.new.tsv.tsv',
                      help='Early relevance predictions: default=%default')
    parser.add_option('--pred-rel-modern', type=str, default='data/speeches/Presidential/predictions/relevance_modern/pred.pres.keywords.new.tsv.tsv',
                      help='Modern relevance predictiosn: default=%default')
    parser.add_option('--pred-tone-early', type=str, default='data/speeches/Presidential/predictions/tone_early/pred.pres.keywords.new.tsv.tsv',
                      help='Early tone predictions: default=%default')
    parser.add_option('--pred-tone-modern', type=str, default='data/speeches/Presidential/predictions/tone_modern/pred.pres.keywords.new.tsv.tsv',
                      help='Modern tone predictiosn: default=%default')
    parser.add_option('--outdir', type=str, default='data/speeches/Congress/',
                      help='Output directory: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    orig_file = options.orig_app_file
    keywords_file = options.pres_keywords_file
    early_rel_file = options.pred_rel_early
    modern_rel_file = options.pred_rel_modern
    early_tone_file = options.pred_tone_early
    modern_tone_file = options.pred_tone_modern
    outdir = options.outdir

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("Reading data")
    with open(orig_file) as f:
        lines = f.readlines()
    documents = [json.loads(line) for line in lines]

    with open(keywords_file) as f:
        lines = f.readlines()
    paragraphs = [json.loads(line) for line in lines]

    speech_years = {}
    for line in documents:
        date = line['date']
        speech_id = line['url']
        parts = date.strip().split()
        year = int(parts[-1])
        assert 1700 < year < 2022
        speech_years[speech_id] = year
    print(len(speech_years))

    all_by_url = {line['url']: line for line in documents}
    print(len(all_by_url))

    print("Loading predictions")
    early_rel_df = pd.read_csv(early_rel_file, header=0, index_col=None, sep='\t')
    modern_rel_df = pd.read_csv(modern_rel_file, header=0, index_col=None, sep='\t')
    early_tone_df = pd.read_csv(early_tone_file, header=0, index_col=None, sep='\t')
    modern_tone_df = pd.read_csv(modern_tone_file, header=0, index_col=None, sep='\t')

    early_probs = df_to_probs(early_rel_df, ['0', '1'])
    modern_probs = df_to_probs(modern_rel_df, ['0', '1'])
    early_tone_probs = df_to_probs(early_tone_df, ['0', '1', '2'])
    modern_tone_probs = df_to_probs(modern_tone_df, ['0', '1', '2'])

    early_first, early_last = get_early_congress_range()
    modern_first, modern_last = get_modern_congress_range()

    valid_cats = {'misc_remarks', 'written_statements', 'written_messages', 'statements',
                  'spoken_addresses', 'press_briefings', 'letters', 'news_conferences',
                  'sotu_messages', 'sotu_addresses', 'inaugurals', 'farewell'}

    print("Gathering segmetns")
    outlines = []
    for i, paragraph in tqdm(enumerate(paragraphs)):
        early_prob = early_probs[i, 1]
        modern_prob = modern_probs[i, 1]
        paragraph_id = paragraph['id']
        tokens = paragraph['tokens']
        sents = []
        for sent in tokens:
            sents.append(' '.join(sent))
        text = ' ' .join(sents)
        speech_url = paragraph_id[:-5]
        person = all_by_url[speech_url]['person']
        title = all_by_url[speech_url]['title']
        date = all_by_url[speech_url]['date']
        year = speech_years[speech_url]
        categories = set(all_by_url[speech_url]['categories'])
        if len(categories.intersection(valid_cats)) > 0:
            if person == 'Grover Cleveland':
                if year < 1892:
                    person = 'Grover Cleveland (85-89)'
                else:
                    person = 'Grover Cleveland (93-97)'
            congress = year_to_congress(year)
            # use prediction based on timing of speech
            if congress <= early_last:
                prob = early_prob
                tone_probs = early_tone_probs[i, :]
            elif early_last < congress < modern_first:
                # interpolate predictions in the middle period
                modern_weight = min(max((congress - early_last) / (modern_first - early_last), 0.), 1.)
                interpolated_prob = (1. - modern_weight) * early_prob + modern_weight * modern_prob
                interpolated_tone_probs = (1. - modern_weight) * early_tone_probs[i, :] + modern_weight * modern_tone_probs[i, :]
                prob = interpolated_prob
                tone_probs = interpolated_tone_probs
            else:
                prob = modern_prob
                tone_probs = modern_tone_probs[i, :]

            # only keep positive predictions fo relevance
            if prob >= 0.5:
                outline = {'id': paragraph_id,
                           'speaker': person,
                           'title': title,
                           'date': date,
                           'year': year,
                           'congress': congress,
                           'categories': sorted(categories),
                           'imm_prob': prob,
                           'anti_prob': tone_probs[0],
                           'neutral_prob': tone_probs[1],
                           'pro_prob': tone_probs[2],
                           'text': text
                           }
                outlines.append(outline)


    print("Saving {:d} lines".format(len(outlines)))
    with open(os.path.join(outdir, 'pres_imm_segments_with_tone.jsonlist'), 'w') as f:
        for line in outlines:
            f.write(json.dumps(line) + '\n')

    df_out = pd.DataFrame()
    columns = ['id', 'speaker', 'title', 'date', 'year', 'congress', 'imm_prob', 'anti_prob', 'neutral_prob', 'pro_prob', 'text']
    for col in columns:
        df_out[col] = [line[col] for line in outlines]
    df_out.to_csv(os.path.join(outdir, 'pres_imm_segments_with_tone.tsv'), sep='\t')


def df_to_probs(df, cols):
    values_exp = np.exp(df[cols].values)
    n_rows, n_cols = values_exp.shape
    probs = values_exp / np.sum(values_exp, axis=1).reshape((n_rows, 1))
    return probs


if __name__ == '__main__':
    main()
