import os
import json
from optparse import OptionParser
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

from relevance.collect_predictions import uscr_date_to_congress


# Same as collect predictions modern, but using linear model predictions


TONES = {0: 'anti', 1: 'neutral', 2: 'pro'}

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--segments-dir', type=str, default='/u/scr/nlp/data/congress/segments/',
                      help='Directory with raw segments: default=%default')
    parser.add_option('--uscr-segments-dir', type=str, default='/u/scr/nlp/data/congress/uscr_segments/',
                      help='Directory with raw segments: default=%default')
    parser.add_option('--keywords-dir', type=str, default='/u/scr/dcard/data/congress/keyword_segments_validation/',
                      help='Directory with model predictions: default=%default')
    #parser.add_option('--early-model-dir', type=str, default='/u/scr/dcard/projects/congressional-annotations/data/congress_early/relevance/splits/label-weights/all/all__s1040749045_lr2e-05_msl512/',
    #                  help='Directory with model predictions: default=%default')
    #parser.add_option('--modern-model-dir', type=str, default='/u/scr/dcard/projects/congressional-annotations/data/congress_modern/relevance/label-weights/all/all__s2405527281_lr2e-05_msl512/',
    #                  help='Directory with model predictions: default=%default')
    #parser.add_option('--metadata-dir', type=str, default='/u/scr/nlp/data/congress/metadata/',
    #                  help='Directory with model predictions: default=%default')
    parser.add_option('--pred-dir', type=str, default='/u/scr/dcard/data/congress/validation/tone/exp/linear/',
                      help='Directory with model predictions: default=%default')
    #parser.add_option('--dates-file', type=str, default='/u/scr/nlp/data/congress/speech_dates.json',
    #                  help='File with speech dates: default=%default')
    #parser.add_option('--uscr-dates-file', type=str, default='/u/scr/nlp/data/congress/uscr_speech_dates.json',
    #                  help='File with speech dates: default=%default')
    parser.add_option('--relevance-dir', type=str, default='/u/scr/dcard/data/congress/validation_linear/',
                      help='Output dir of relevance.collect_predictions_linear: default=%default')
    
    #parser.add_option('--relevance-file', type=str, default='/u/scr/dcard/data/congress/validation_linear/imm_speech_ids_all.tsv',
    #                  help='Output of relevance.export_presidential_segments.py: default=%default')
    #parser.add_option('--keywords-segment-probs', type=str, default='/u/scr/dcard/data/congress/validation_linear/keyword_segment_probs_selected.json',
    #                  help='Output of relevance.export_presidential_segments.py: default=%default')
    #parser.add_option('--non-keywords-segment-probs', type=str, default='/u/scr/dcard/data/congress/validation_linear/non_keyword_segment_probs.json',
    #                  help='Output of relevance.export_presidential_segments.py: default=%default')
    parser.add_option('--uscr-transition', type=int, default=112,
                      help='The Congress at which to start using USCR instead of Gentzkow: default=%default')
    parser.add_option('--outdir', type=str, default='/u/scr/dcard/data/congress/validation_linear/',
                      help='Outdir: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    print(options)

    segments_dir = options.segments_dir
    uscr_segments_dir = options.uscr_segments_dir
    keywords_dir = options.keywords_dir
    #early_model_dir = options.early_model_dir
    #modern_model_dir = options.modern_model_dir
    #metadata_dir = options.metadata_dir
    pred_dir = options.pred_dir
    #speech_dates_file = options.dates_file
    #uscr_speech_dates_file = options.uscr_dates_file
    
    relevance_dir = options.relevance_dir    
    relevance_file = os.path.join(relevance_dir, 'imm_speech_ids_all.tsv')
    keywords_segment_probs_file = os.path.join(relevance_dir, 'keyword_segment_probs_selected.json')
    non_keywords_segment_probs_file = os.path.join(relevance_dir, 'non_keyword_segment_probs.json')
    uscr_transition = options.uscr_transition
    outdir = options.outdir

    early_first = 43
    uscr_last = 116

    print("Loading relevance data")
    print(relevance_file)
    df = pd.read_csv(relevance_file, header=0, index_col=0, sep='\t')
    imm_speech_ids = ([str(i) for i in df['speech_id'].values])
    imm_speech_id_set = set(imm_speech_ids)

    dates = df['date'].values
    congresses = df['congress'].values
    chambers = df['chamber'].values
    states = df['state'].values
    speakers = df['speaker'].values
    
    dates_by_id = dict(zip(imm_speech_ids, dates))
    congresses_by_id = dict(zip(imm_speech_ids, congresses))
    chambers_by_id = dict(zip(imm_speech_ids, chambers))
    states_by_id = dict(zip(imm_speech_ids, states))
    speakers_by_id = dict(zip(imm_speech_ids, speakers))

    """
    print("Loading speech dates")
    with open(speech_dates_file) as f:
        speech_dates = json.load(f)
    with open(uscr_speech_dates_file) as f:
        uscr_speech_dates = json.load(f)
    all_speech_dates = {}
    all_speech_dates.update(speech_dates)
    all_speech_dates.update(uscr_speech_dates)
    """

    with open(keywords_segment_probs_file) as f:
        keywords_segment_probs = json.load(f)

    with open(non_keywords_segment_probs_file) as f:
        non_keywords_segment_probs = json.load(f)

    print("Loading keyword data")
    keyword_segments_file = os.path.join(keywords_dir, 'keyword_segments_all.jsonlist')

    keyword_pred_file = os.path.join(pred_dir, 'pred.probs.keyword_segments_all.csv')

    print("Loading keyword segments")
    keyword_segments = load_segments(keyword_segments_file)
    print("Loaded {:d} segments".format(len(keyword_segments)))

    lines = []
    for line in keyword_segments:
        if line['id'] in keywords_segment_probs:
            lines.append(line)
    print("Found {:d} relevant segments".format(len(lines)))    
    keyword_segments = lines
     
    print("Loading keyword preds")
    keyword_probs = load_pred_probs(keyword_pred_file)

    print("Counting speeches and getting max probs")
    keyword_label_counts = defaultdict(Counter)
    keyword_prob_sums = defaultdict(Counter)
    labeled_speech_dict = defaultdict(dict)
    keyword_label_counts, keyword_prob_sums, keywords_segment_probs, labeled_speech_dict = update_keyword_speech_probs(keyword_label_counts, keyword_prob_sums, keywords_segment_probs, keyword_segments, keyword_probs, early_first, uscr_last, imm_speech_id_set, labeled_speech_dict)
    #keyword_label_counts, keyword_prob_sums, keywords_segment_probs = update_keyword_speech_probs(keyword_label_counts, keyword_prob_sums, keywords_segment_probs, mid_keyword_segments, mid_keyword_probs, early_last+1, modern_first-1, imm_speech_id_set)
    #keyword_label_counts, keyword_prob_sums, keywords_segment_probs = update_keyword_speech_probs(keyword_label_counts, keyword_prob_sums, keywords_segment_probs, modern_keyword_segments, modern_keyword_probs, modern_first, uscr_transition-1, imm_speech_id_set)
    #keyword_label_counts, keyword_prob_sums, keywords_segment_probs = update_keyword_speech_probs(keyword_label_counts, keyword_prob_sums, keywords_segment_probs, uscr_keyword_segments, uscr_keyword_probs, uscr_transition, uscr_last, imm_speech_id_set, uscr=True)

    # add non-keyword speeches for each congress
    all_label_counts = defaultdict(Counter)
    all_prob_sums = defaultdict(Counter)

    for uscr in [0, 1]:
        if uscr:
            start = uscr_transition
            end = uscr_last
        else:
            start = early_first
            end = uscr_transition-1
        for congress in range(start, end+1):
            count = 0
            if uscr:
                segments_file = os.path.join(uscr_segments_dir, 'segments-' + str(congress).zfill(3) + '.jsonlist')
            else:
                segments_file = os.path.join(segments_dir, 'segments-' + str(congress).zfill(3) + '.jsonlist')
            pred_file = os.path.join(pred_dir, 'pred.probs.segments-' + str(congress).zfill(3) + '.csv')
            if not os.path.exists(pred_file):
                print("Skipping", segments_file)
            else:
                segments = load_segments(segments_file)
                pred_probs = load_pred_probs(pred_file)

                print("Loaded {:d} segments".format(len(segments)))

                lines = []
                for line in segments:
                    if line['id'] in non_keywords_segment_probs:
                        lines.append(line)
                print("Found {:d} relevant segments".format(len(lines)))    
                segments = lines

                try:
                    assert len(pred_probs) == len(segments)
                except AssertionError as e:
                    print(len(pred_probs))
                    print(len(segments))
                    raise e

                pred_labels = np.argmax(pred_probs, axis=1)

                for line_i, line in enumerate(segments):
                    segment_id = line['id']
                    parts = segment_id.split('_')
                    if uscr:
                        speech_id = '_'.join(parts[:2])
                    else:
                        speech_id = parts[0]

                    if segment_id in non_keywords_segment_probs:
                        assert speech_id in imm_speech_id_set
                        pred_label = pred_labels[line_i]
                        all_label_counts[speech_id][pred_label] += 1
                        for tone in range(3):
                            all_prob_sums[speech_id][tone] += pred_probs[line_i, tone]
                            non_keywords_segment_probs[segment_id][TONES[tone]] = pred_probs[line_i, tone]
                        count += 1
                        if 'non_keyword_probs' in labeled_speech_dict[speech_id]:
                            labeled_speech_dict[speech_id]['non_keyword_probs'].append(list([float(v) for v in pred_probs[line_i]]))
                        else:
                            labeled_speech_dict[speech_id]['non_keyword_probs'] = [list([float(v) for v in pred_probs[line_i]])]

                print(uscr, congress, count, len(all_label_counts))

    prob_overall_sum = np.zeros(3)
    for speech_id, probs in all_prob_sums.items():
        for p, v in probs.items():
            prob_overall_sum[p] += v

    prob_overall_dist = prob_overall_sum / prob_overall_sum.sum()
    print(prob_overall_dist)

    # get label count and speech probes for the keyword speeches
    df['keyword_anti_count'] = [keyword_label_counts[i][0] for i in imm_speech_ids]
    df['keyword_neutral_count'] = [keyword_label_counts[i][1] for i in imm_speech_ids]
    df['keyword_pro_count'] = [keyword_label_counts[i][2] for i in imm_speech_ids]

    keyword_anti_probs = np.array([keyword_prob_sums[i][0] for i in imm_speech_ids])
    keyword_netural_probs = np.array([keyword_prob_sums[i][1] for i in imm_speech_ids])
    keyword_pro_probs = np.array([keyword_prob_sums[i][2] for i in imm_speech_ids])

    df['keyword_anti_prob_sums'] = keyword_anti_probs
    df['keyword_neutral_prob_sums'] = keyword_netural_probs
    df['keyword_pro_prob_sums'] = keyword_pro_probs

    # normalize with slight smoothing
    df['keyword_tone'] = (keyword_pro_probs - keyword_anti_probs) / (keyword_anti_probs + keyword_netural_probs + keyword_pro_probs + 1)

    df['nonkeyword_anti_count'] = [all_label_counts[i][0] for i in imm_speech_ids]
    df['nonkeyword_neutral_count'] = [all_label_counts[i][1] for i in imm_speech_ids]
    df['nonkeyword_pro_count'] = [all_label_counts[i][2] for i in imm_speech_ids]

    nonkeyword_anti_probs = np.array([all_prob_sums[i][0] for i in imm_speech_ids])
    nonkeyword_netural_probs = np.array([all_prob_sums[i][1] for i in imm_speech_ids])
    nonkeyword_pro_probs = np.array([all_prob_sums[i][2] for i in imm_speech_ids])

    df['nonkeyword_anti_prob_sums'] = nonkeyword_anti_probs
    df['nonkeyword_neutral_prob_sums'] = nonkeyword_netural_probs
    df['nonkeyword_pro_prob_sums'] = nonkeyword_pro_probs

    df['nonkeyword_tone'] = (nonkeyword_pro_probs - nonkeyword_anti_probs) / (nonkeyword_anti_probs + nonkeyword_netural_probs + nonkeyword_pro_probs + 1)

    keyword_indicator = np.array(df['keyword'].values)
    overall_tone = keyword_indicator * np.array(df['keyword_tone'].values) + (1-keyword_indicator) * np.array(df['nonkeyword_tone'].values)
    overall_pro_sum = keyword_indicator * np.array(df['keyword_pro_prob_sums'].values) + (1-keyword_indicator) * np.array(df['nonkeyword_pro_prob_sums'].values)
    overall_neutral_sum = keyword_indicator * np.array(df['keyword_neutral_prob_sums'].values) + (1-keyword_indicator) * np.array(df['nonkeyword_neutral_prob_sums'].values)
    overall_anti_sum = keyword_indicator * np.array(df['keyword_anti_prob_sums'].values) + (1-keyword_indicator) * np.array(df['nonkeyword_anti_prob_sums'].values)

    df['pro_prob_sum'] = overall_pro_sum
    df['neutral_prob_sum'] = overall_neutral_sum
    df['anti_prob_sum'] = overall_anti_sum
    df['tone'] = overall_tone

    print(df.shape)
    outfile = os.path.join(outdir, 'imm_speech_ids_with_tone.tsv')
    df.to_csv(outfile, sep='\t')

    outfile = os.path.join(outdir, 'keyword_segment_probs_selected_with_tone.json')
    with open(outfile, 'w') as f:
        json.dump(keywords_segment_probs, f, indent=2)

    outfile = os.path.join(outdir, 'non_keyword_segment_probs_with_tone.json')
    with open(outfile, 'w') as f:
        json.dump(non_keywords_segment_probs, f, indent=2)

    for speech_id, values in labeled_speech_dict.items():
        values['date'] = int(dates_by_id[speech_id])
        values['congress'] = int(congresses_by_id[speech_id])
        values['chamber'] = chambers_by_id[speech_id]
        values['state'] = states_by_id[speech_id]
        values['speaker'] = speakers_by_id[speech_id]

    outfile = os.path.join(outdir, 'tone_by_speech_id.json')
    with open(outfile, 'w') as f:
        json.dump(labeled_speech_dict, f, indent=2)


def load_segments(infile):
    segments = []
    with open(infile) as f:
        for line in f:
            segments.append(json.loads(line))
    return segments


def load_pred_probs(infile):
    df = pd.read_csv(infile, header=0, index_col=0, sep=',')
    #df.columns = ['predicted', 'anti', 'neutral', 'pro']
    #pred_labels = df['predicted'].values
    pred_scores_exp = np.exp(df[['anti', 'neutral', 'pro']].values)
    n_items, n_cats = pred_scores_exp.shape
    pred_probs = pred_scores_exp / pred_scores_exp.sum(1).reshape((n_items, 1))
    #yes_probs = pred_probs[:, 1]
    #assert (pred_labels == np.argmax(pred_probs, axis=1)).all()
    return np.array(pred_probs)


def update_keyword_speech_probs(keyword_label_counts, keyword_prob_sums, keywords_segment_probs, segments, pred_probs, first_congress, last_congress, imm_speech_id_set, labeled_speech_dict):
    try:
        assert len(segments) == len(pred_probs)
    except AssertionError as e:
        print(len(segments))
        print(len(pred_probs))
        raise e
    pred_labels = np.argmax(pred_probs, axis=1)

    for line_i, line in enumerate(segments):
        segment_id = line['id']
        parts = segment_id.split('_')
        if segment_id.startswith('C'):
            speech_id = '_'.join(parts[:2])
            parts2 = speech_id.split('-')
            year = int(parts2[1])
            month = int(parts2[2])
            day = int(parts2[3])
            congress = uscr_date_to_congress(year, month, day)
        else:
            speech_id = parts[0]
            if speech_id.startswith('1'):
                congress = int(speech_id[:3])
            else:
                congress = int(speech_id[:2])
        if first_congress <= congress <= last_congress and speech_id != 'speech':
            #if speech_id in imm_speech_id_set:
            if segment_id in keywords_segment_probs:
                assert speech_id in imm_speech_id_set
                pred_label = pred_labels[line_i]
                keyword_label_counts[speech_id][pred_label] += 1
                # combine segment scores into speech scores (as sums)
                for tone in range(3):
                    keyword_prob_sums[speech_id][tone] += pred_probs[line_i, tone]
                    keywords_segment_probs[segment_id][TONES[tone]] = pred_probs[line_i, tone]
                if speech_id in labeled_speech_dict:
                    labeled_speech_dict[speech_id]['keyword_probs'].append(list([float(v) for v in pred_probs[line_i]]))
                else:
                    labeled_speech_dict[speech_id]['keyword_probs'] = [list([float(v) for v in pred_probs[line_i]])]

    return keyword_label_counts, keyword_prob_sums, keywords_segment_probs, labeled_speech_dict


if __name__ == '__main__':
    main()
