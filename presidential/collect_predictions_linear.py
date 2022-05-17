import os
import json
from optparse import OptionParser
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
from time_periods.common import year_to_congress


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--orig-app-file', type=str, default='/<path_to_scrapers>/data/app/all.jsonlist',
                      help='Original file with all documents from scrapers/app: default=%default')
    parser.add_option('--pres-keywords-file', type=str, default='data/speeches/Presidential/paragraphs.keywords.combo.jsonlist',
                      help='Keyword paragraphs file: default=%default')
    parser.add_option('--pred-rel', type=str, default='data/speeches/Presidential/predictions/relevance_linear/pred.probs.paragraphs.keywords.combo.csv',
                      help='relevance predictions: default=%default')
    parser.add_option('--pred-tone', type=str, default='data/speeches/Presidential/predictions/tone_linear/pred.probs.paragraphs.keywords.combo.csv',
                      help='tone predictions: default=%default')
    parser.add_option('--outdir', type=str, default='/data/speeches/Presidential/predictions/',
                      help='Output directory: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    orig_file = options.orig_app_file
    keywords_file = options.pres_keywords_file
    rel_file = options.pred_rel
    tone_file = options.pred_tone
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
    rel_df = pd.read_csv(rel_file, header=0, index_col=0)
    tone_df = pd.read_csv(tone_file, header=0, index_col=0)

    pred_scores_exp = np.exp(rel_df[['no', 'yes']].values)
    n_items, n_cats = pred_scores_exp.shape
    pred_probs = pred_scores_exp / pred_scores_exp.sum(1).reshape((n_items, 1))
    yes_probs_all = pred_probs[:, 1]

    pred_scores_exp = np.exp(tone_df[['anti', 'neutral', 'pro']].values)
    n_items, n_cats = pred_scores_exp.shape
    pred_probs = pred_scores_exp / pred_scores_exp.sum(1).reshape((n_items, 1))
    tone_probs_all = np.array(pred_probs)

    valid_cats = {'misc_remarks', 'written_statements', 'written_messages', 'statements',
                  'spoken_addresses', 'press_briefings', 'letters', 'news_conferences',
                  'sotu_messages', 'sotu_addresses', 'inaugurals', 'farewell'}

    print("Gathering segmetns")
    outlines = []
    for i, paragraph in tqdm(enumerate(paragraphs)):
        yes_prob = yes_probs_all[i]
        tone_probs = tone_probs_all[i]
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

            # only keep positive predictions fo relevance
            if yes_prob >= 0.5:
                outline = {'id': paragraph_id,
                           'speaker': person,
                           'title': title,
                           'date': date,
                           'year': year,
                           'congress': congress,
                           'categories': sorted(categories),
                           'imm_prob': yes_prob,
                           'anti_prob': tone_probs[0],
                           'neutral_prob': tone_probs[1],
                           'pro_prob': tone_probs[2],
                           'text': text
                           }
                outlines.append(outline)


    print("Saving {:d} lines".format(len(outlines)))
    with open(os.path.join(outdir, 'pres_imm_segments_with_tone_linear.jsonlist'), 'w') as f:
        for line in outlines:
            f.write(json.dumps(line) + '\n')

    df_out = pd.DataFrame()
    columns = ['id', 'speaker', 'title', 'date', 'year', 'congress', 'imm_prob', 'anti_prob', 'neutral_prob', 'pro_prob', 'text']
    for col in columns:
        df_out[col] = [line[col] for line in outlines]
    df_out.to_csv(os.path.join(outdir, 'pres_imm_segments_with_tone_linear.tsv'), sep='\t')



if __name__ == '__main__':
    main()
