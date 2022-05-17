import os
import json
from glob import glob
from optparse import OptionParser
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from relevance.collect_predictions import uscr_date_to_congress
from time_periods.common import year_to_congress


# Alternative to tone/collect_prediction.py, to be used for the validation
# in SI using the Bayesian correcction, with matrices estimated from error rates


TONES = {0: 'anti', 1: 'neutral', 2: 'pro'}
TONE_INDEX = {'anti': 0, 'neutral': 1, 'pro': 2}

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--segments-dir', type=str, default='data/speeches/Congress/segments/',
                      help='Directory with raw segments: default=%default')
    parser.add_option('--uscr-segments-dir', type=str, default='data/speeches/Congress/uscr_segments/',
                      help='Directory with raw segments: default=%default')
    parser.add_option('--keywords-dir', type=str, default='data/speeches/Congress/keyword_segments/',
                      help='Directory with model predictions: default=%default')
    parser.add_option('--early-model-dir', type=str, default='data/speeches/Congress/data/congress_early/tone/splits/label-weights/all/all__s1040749045_lr2e-05_msl512/',
                      help='Directory with model predictions: default=%default')
    parser.add_option('--modern-model-dir', type=str, default='data/speeches/Congress/data/congress_modern/tone_extra/label-weights/all/all__s2405527281_lr2e-05_msl512/',
                      help='Directory with model predictions: default=%default')
    parser.add_option('--metadata-dir', type=str, default='data/speeches/Congress/metadata/',
                      help='Directory with model predictions: default=%default')
    parser.add_option('--relevance-file', type=str, default='data/speeches/Congress/imm_speech_ids_all.tsv',
                      help='Output of relevance.export_presidential_segments.py: default=%default')
    parser.add_option('--keywords-segment-probs', type=str, default='data/speeches/Congress/keyword_segment_probs_selected.json',
                      help='Output of relevance.export_presidential_segments.py: default=%default')
    parser.add_option('--non-keywords-segment-probs', type=str, default='data/speeches/Congress/non_keyword_segment_probs.json',
                      help='Output of relevance.export_presidential_segments.py: default=%default')
    parser.add_option('--uscr-transition', type=int, default=112,
                      help='The Congress at which to start using USCR instead of Gentzkow: default=%default')
    parser.add_option('--cfm-window', type=int, default=10,
                      help='Number of congresses to consider on either side: default=%default')
    parser.add_option('--inferred-label-dir', type=str, default='data/annotations/relevance_and_tone/inferred_labels/',
                      help='Inferred label dir (for cfm correction): default=%default')
    parser.add_option('--party-cfm', type=str, default='na',
                      help='Party for correction (R or D): default=%default')
    parser.add_option('--outdir', type=str, default='data/speeches/Congress/predictions_cfm',
                      help='Outdir: default=%default')

    (options, args) = parser.parse_args()

    segments_dir = options.segments_dir
    uscr_segments_dir = options.uscr_segments_dir
    keywords_dir = options.keywords_dir
    early_model_dir = options.early_model_dir
    modern_model_dir = options.modern_model_dir
    metadata_dir = options.metadata_dir
    relevance_file = options.relevance_file
    keywords_segment_probs_file = options.keywords_segment_probs
    non_keywords_segment_probs_file = options.non_keywords_segment_probs
    uscr_transition = options.uscr_transition
    cfm_correction = True
    cfm_window = options.cfm_window
    inferred_label_dir = options.inferred_label_dir
    party_cfm = options.party_cfm
    outdir = options.outdir

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    early_first = 43
    early_last = 73
    mid_first = 70
    mid_last = 88
    modern_first = 85
    modern_last = 114
    uscr_first = 104
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

    with open(keywords_segment_probs_file) as f:
        keywords_segment_probs = json.load(f)

    with open(non_keywords_segment_probs_file) as f:
        non_keywords_segment_probs = json.load(f)

    print("Loading keyword data")
    early_keywords_segments_file = os.path.join(keywords_dir, 'keyword_segments_' + str(early_first) + '-' + str(early_last) + '.jsonlist')
    mid_keywords_segments_file = os.path.join(keywords_dir, 'keyword_segments_' + str(mid_first) + '-' + str(mid_last) + '.jsonlist')
    modern_keywords_segments_file = os.path.join(keywords_dir, 'keyword_segments_' + str(modern_first) + '-' + str(modern_last) + '.jsonlist')
    uscr_keywords_segments_file = os.path.join(keywords_dir, 'keyword_segments_uscr_' + str(uscr_first) + '-' + str(uscr_last) + '.jsonlist')

    early_keywords_pred_file = os.path.join(early_model_dir, 'pred.keywords-' + str(early_first) + '-' + str(early_last) + '.new.tsv.tsv')
    early_mid_keywords_pred_file = os.path.join(early_model_dir, 'pred.keywords-' + str(mid_first) + '-' + str(mid_last) + '.new.tsv.tsv')
    modern_mid_keywords_pred_file = os.path.join(modern_model_dir, 'pred.keywords-' + str(mid_first) + '-' + str(mid_last) + '.new.tsv.tsv')
    modern_keywords_pred_file = os.path.join(modern_model_dir, 'pred.keywords-' + str(modern_first) + '-' + str(modern_last) + '.new.tsv.tsv')
    uscr_keywords_pred_file = os.path.join(modern_model_dir, 'pred.keywords_uscr-' + str(uscr_first) + '-' + str(uscr_last) + '.new.tsv.tsv')

    print("Loading keyword segments")
    early_keyword_segments = load_segments(early_keywords_segments_file)
    mid_keyword_segments = load_segments(mid_keywords_segments_file)
    modern_keyword_segments = load_segments(modern_keywords_segments_file)
    uscr_keyword_segments = load_segments(uscr_keywords_segments_file)

    print("Loading keyword preds")
    early_keyword_probs = load_pred_probs(early_keywords_pred_file)
    early_mid_keyword_probs = load_pred_probs(early_mid_keywords_pred_file)
    modern_mid_keyword_probs = load_pred_probs(modern_mid_keywords_pred_file)
    modern_keyword_probs = load_pred_probs(modern_keywords_pred_file)
    uscr_keyword_probs = load_pred_probs(uscr_keywords_pred_file)

    print("Loading metadata")
    id_to_party = {}
    id_to_congress = {}
    files = sorted(glob(os.path.join(metadata_dir, '*metadata_*.json')))
    for infile in tqdm(files):
        with open(infile) as f:
            data = json.load(f)
        for key, vals in data.items():
            party = vals['party']
            year = int(vals['year'])
            congress = year_to_congress(year)
            id_to_party[key] = party
            id_to_congress[key] = congress

    early_keyword_congresses = [id_to_congress['_'.join(line['id'].split('_')[:-1])] for line in early_keyword_segments]
    mid_keyword_congresses = [id_to_congress['_'.join(line['id'].split('_')[:-1])] for line in mid_keyword_segments]
    modern_keyword_congresses = [id_to_congress['_'.join(line['id'].split('_')[:-1])] for line in modern_keyword_segments]
    uscr_keyword_congresses = [id_to_congress['_'.join(line['id'].split('_')[:-1])] for line in uscr_keyword_segments]

    if cfm_correction:
        tone_counter_by_congress = defaultdict(Counter)
        for data_period in ['early', 'mid', 'modern']:
            infile = os.path.join(inferred_label_dir, data_period + '_tone_all.jsonlist')
            with open(infile) as f:
                lines = f.readlines()
            lines = [json.loads(line) for line in lines]

            for line in lines:
                speech_id = line['id'].split('_')[0]
                if speech_id in id_to_party:
                    party = id_to_party[speech_id]
                    congress = id_to_congress[speech_id]
                    if party_cfm == 'na' or party == party_cfm:
                        for c in range(congress-cfm_window, congress+cfm_window+1):
                            tone_counter_by_congress[c][line['label']] += 1

        priors_by_congress = {}
        for c, counter in tone_counter_by_congress.items():
            prior = np.array([counter['anti'], counter['neutral'], counter['pro']]) / sum(counter.values())            
            priors_by_congress[c] = prior

        for c, prior in priors_by_congress.items():
            print(c, prior)

        early_keyword_probs = apply_cfm_correction(early_keyword_probs, early_keyword_congresses, priors_by_congress, 'early', 'early')
        early_mid_keyword_probs = apply_cfm_correction(early_mid_keyword_probs, mid_keyword_congresses, priors_by_congress, 'early', 'mid')
        modern_mid_keyword_probs = apply_cfm_correction(modern_mid_keyword_probs, mid_keyword_congresses, priors_by_congress, 'modern', 'mid', party_cfm)
        modern_keyword_probs = apply_cfm_correction(modern_keyword_probs, modern_keyword_congresses, priors_by_congress, 'modern', 'modern', party_cfm)
        uscr_keyword_probs = apply_cfm_correction(uscr_keyword_probs, uscr_keyword_congresses, priors_by_congress, 'modern', 'modern', party_cfm)


    mid_keyword_probs = interpolate_mid_probs(mid_keyword_segments, early_mid_keyword_probs, modern_mid_keyword_probs, early_last, modern_first)

    print("Counting speeches and getting max probs")
    keyword_label_counts = defaultdict(Counter)
    keyword_prob_sums = defaultdict(Counter)
    labeled_speech_dict = defaultdict(dict)
    
    keyword_label_counts, keyword_prob_sums, keywords_segment_probs, labeled_speech_dict = update_keyword_speech_probs(keyword_label_counts, keyword_prob_sums, keywords_segment_probs, early_keyword_segments, early_keyword_probs, early_first, early_last, imm_speech_id_set, labeled_speech_dict)
    keyword_label_counts, keyword_prob_sums, keywords_segment_probs, labeled_speech_dict = update_keyword_speech_probs(keyword_label_counts, keyword_prob_sums, keywords_segment_probs, mid_keyword_segments, mid_keyword_probs, early_last+1, modern_first-1, imm_speech_id_set, labeled_speech_dict)
    keyword_label_counts, keyword_prob_sums, keywords_segment_probs, labeled_speech_dict = update_keyword_speech_probs(keyword_label_counts, keyword_prob_sums, keywords_segment_probs, modern_keyword_segments, modern_keyword_probs, modern_first, uscr_transition-1, imm_speech_id_set, labeled_speech_dict)
    keyword_label_counts, keyword_prob_sums, keywords_segment_probs, labeled_speech_dict = update_keyword_speech_probs(keyword_label_counts, keyword_prob_sums, keywords_segment_probs, uscr_keyword_segments, uscr_keyword_probs, uscr_transition, uscr_last, imm_speech_id_set, labeled_speech_dict, uscr=True)

    keyword_prob_overall_sum = np.zeros(3)
    for speech_id, probs in keyword_prob_sums.items():
        for p, v in probs.items():
            keyword_prob_overall_sum[p] += v
    #keyword_prob_overall_dist = keyword_prob_overall_sum / keyword_prob_overall_sum.sum()

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
            segments = load_segments(segments_file)
            if uscr:
                pred_file = os.path.join(modern_model_dir, 'pred.segments.uscr.new-' + str(congress).zfill(3) + '.tsv.tsv')
                pred_probs = load_pred_probs(pred_file)
                if cfm_correction:
                    pred_probs = apply_cfm_correction(pred_probs, [congress] * len(pred_probs), priors_by_congress, 'modern', 'modern', party_cfm)
            elif congress <= early_last:
                pred_file = os.path.join(early_model_dir, 'pred.segments-' + str(congress).zfill(3) + '.tsv.tsv')
                pred_probs = load_pred_probs(pred_file)
                if cfm_correction:
                    pred_probs = apply_cfm_correction(pred_probs, [congress] * len(pred_probs), priors_by_congress, 'early', 'early')
            elif congress >= modern_first:
                pred_file = os.path.join(modern_model_dir, 'pred.segments-' + str(congress).zfill(3) + '.tsv.tsv')
                pred_probs = load_pred_probs(pred_file)
                if cfm_correction:
                    pred_probs = apply_cfm_correction(pred_probs, [congress] * len(pred_probs), priors_by_congress, 'modern', 'modern', party_cfm)
            else:
                early_pred_file = os.path.join(early_model_dir, 'pred.segments-' + str(congress).zfill(3) + '.tsv.tsv')
                early_tone_probs = load_pred_probs(early_pred_file)
                if cfm_correction:
                    early_tone_probs = apply_cfm_correction(early_tone_probs, [congress] * len(early_tone_probs), priors_by_congress, 'early', 'mid')
                modern_pred_file = os.path.join(modern_model_dir, 'pred.segments-' + str(congress).zfill(3) + '.tsv.tsv')
                modern_tone_probs = load_pred_probs(modern_pred_file)
                if cfm_correction:
                    modern_tone_probs = apply_cfm_correction(modern_tone_probs, [congress] * len(modern_tone_probs), priors_by_congress, 'modern', 'mid', party_cfm)
                modern_weight = min(max((congress - early_last) / (modern_first - early_last), 0.), 1.)
                pred_probs = (1. - modern_weight) * early_tone_probs + modern_weight * modern_tone_probs
            assert len(pred_probs) == len(segments)

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
    df = pd.read_csv(infile, header=0, index_col=None, sep='\t')
    df.columns = ['predicted', 'anti', 'neutral', 'pro']
    pred_labels = df['predicted'].values
    pred_scores_exp = np.exp(df[['anti', 'neutral', 'pro']].values)
    n_items, n_cats = pred_scores_exp.shape
    pred_probs = pred_scores_exp / pred_scores_exp.sum(1).reshape((n_items, 1))
    #yes_probs = pred_probs[:, 1]
    assert (pred_labels == np.argmax(pred_probs, axis=1)).all()
    return np.array(pred_probs)


def interpolate_mid_probs(segments, early_probs, modern_probs, early_last, modern_first):
    interpolated_probs = []
    for line_i, line in enumerate(segments):
        segment_id = line['id']
        parts = segment_id.split('_')
        speech_id = parts[0]
        congress = int(speech_id[:2])
        modern_weight = min(max((congress - early_last) / (modern_first - early_last), 0.), 1.)
        interpolated_probs.append((1. - modern_weight) * early_probs[line_i] + modern_weight * modern_probs[line_i])
    interpolated_probs = np.vstack(interpolated_probs)
    print(interpolated_probs.shape)
    return interpolated_probs


def update_keyword_speech_probs(keyword_label_counts, keyword_prob_sums, keywords_segment_probs, segments, pred_probs, first_congress, last_congress, imm_speech_id_set, labeled_speech_dict, uscr=False):
    assert len(segments) == len(pred_probs)
    pred_labels = np.argmax(pred_probs, axis=1)

    for line_i, line in enumerate(segments):
        segment_id = line['id']
        parts = segment_id.split('_')
        if uscr:
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


def apply_cfm_correction(pred_probs, congresses, priors_by_congress, model_period='early', data_period='mid', party='na'):

    if model_period == 'early' and data_period == 'early':
        cfm = np.vstack([[0.67296512, 0.25306848, 0.07396641],
                        [0.25547352, 0.63874745, 0.10577902],
                        [0.27254098, 0.2295082,  0.49795082]]
                       )
    elif model_period == 'early' and data_period == 'mid':
        cfm = np.vstack([[0.72093023, 0.24418605, 0.03488372],
                         [0.23076923, 0.67420814, 0.09502262],
                         [0.28125,    0.3125,     0.40625   ]]
                       )
    elif party == 'D' and model_period == 'modern' and data_period == 'mid':                       
        cfm = np.vstack([[0.65979381, 0.24742268, 0.09278351],
                         [0.21428571, 0.64285714, 0.14285714],
                         [0.21153846, 0.25,       0.53846154]]
                       )
    elif party == 'R' and model_period == 'modern' and data_period == 'mid':                       
        cfm = np.vstack([[0.57142857, 0.375,      0.05357143],
                         [0.25373134, 0.71641791, 0.02985075],
                         [0.05263158, 0.34210526, 0.60526316]]
                       )
    elif model_period == 'modern' and data_period == 'mid':
        cfm = np.vstack([[0.63953488, 0.28488372, 0.0755814 ],
                         [0.20361991, 0.69683258, 0.09954751],
                         [0.109375,   0.28125,    0.609375  ]]
                       )        
    elif party == 'D' and model_period == 'modern' and data_period == 'modern':
        cfm = np.vstack([[0.6044686,  0.2807971,  0.1147343 ],
                         [0.14846416, 0.59982935, 0.25170648],
                         [0.06940407, 0.15515988, 0.77543605]]
                        )       
    elif party == 'R' and model_period == 'modern' and data_period == 'modern':
        cfm = np.vstack([[0.72163121, 0.23167849, 0.04669031],
                         [0.25114679, 0.55963303, 0.18922018],
                         [0.0994152,  0.29385965, 0.60672515]]
                        )       
    elif model_period == 'modern':
        cfm = np.vstack([[0.67923851, 0.24335489, 0.07740661],
                         [0.18128059, 0.5632137,  0.25550571],
                         [0.06874096, 0.17510854, 0.75615051]]
                        )       
    else:
        raise ValueError("model_period not recognized") 

    inverse_cfm_by_congress = {}
    
    congress_set = sorted(set(congresses))
    for congress in congress_set:
        inverse_cfm = np.zeros([3, 3])
        for true in range(3):
            for pred in range(3):
                inverse_cfm[pred, true] = cfm[true, pred] * priors_by_congress[congress][true]
        inverse_cfm = inverse_cfm / inverse_cfm.sum(1).reshape((3, 1))
        print(congress, inverse_cfm)
        inverse_cfm_by_congress[congress] = inverse_cfm

    n_rows, n_cols = pred_probs.shape
    outrows = []
    for i in range(n_rows):
        preds = pred_probs[i, :]
        corrected = np.zeros(3)
        for tone in range(3):
            corrected += inverse_cfm_by_congress[congresses[i]][tone, :] * preds[tone]        
        outrows.append(corrected)

    return np.vstack(outrows)


if __name__ == '__main__':
    main()
