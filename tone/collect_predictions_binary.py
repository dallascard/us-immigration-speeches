import os
import json
from glob import glob
from optparse import OptionParser
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from relevance.collect_predictions import uscr_date_to_congress


TONES = {0: 'anti', 1: 'pro'}

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--segments-dir', type=str, default='data/speeches/Congress/segments/',
                      help='Directory with raw segments: default=%default')
    parser.add_option('--uscr-segments-dir', type=str, default='data/speeches/Congress/uscr_segments/',
                      help='Directory with raw segments: default=%default')
    parser.add_option('--keywords-dir', type=str, default='data/speeches/Congress/keyword_segments/',
                      help='Directory with model predictions: default=%default')
    parser.add_option('--segments-pred-dir', type=str, default='data/speeches/Congress/segments_val_tone_binary_preds/',
                      help='Directory with model predictions on keyword segments: default=%default')
    parser.add_option('--metadata-dir', type=str, default='data/speeches/Congress/metadata/',
                      help='Metadata directory: default=%default')
    parser.add_option('--relevance-file', type=str, default='data/speeches/Congress/imm_speech_ids_all.tsv',
                      help='Output of relevance.collect_predictions_val.py: default=%default')
    parser.add_option('--keywords-segment-probs', type=str, default='data/speeches/Congress/keyword_segment_probs_selected.json',
                      help='Output : default=%default')
    parser.add_option('--non-keywords-segment-probs', type=str, default='data/speeches/Congress/non_keyword_segment_probs.json',
                      help='Output : default=%default')
    parser.add_option('--uscr-transition', type=int, default=112,
                      help='The Congress at which to start using USCR instead of Gentzkow: default=%default')
    parser.add_option('--cfm-correction', action="store_true", default=False,
                      help='Use CFM to correct for label bias: default=%default')
    parser.add_option('--inferred-label-dir', type=str, default=None,
                      help='Inferred label dir (for cfm correction): default=%default')
    parser.add_option('--party-cfm', type=str, default='na',
                      help='Party for correction [HACK] (R or D): default=%default')
    parser.add_option('--outdir', type=str, default='data/speeches/Congress/predictions_binary/',
                      help='Outdir: default=%default')

    (options, args) = parser.parse_args()

    print(options)

    segments_dir = options.segments_dir
    uscr_segments_dir = options.uscr_segments_dir
    keywords_dir = options.keywords_dir
    #early_pred_dir = options.early_pred_dir
    #modern_pred_dir = options.modern_pred_dir
    segments_pred_dir = options.segments_pred_dir
    metadata_dir = options.metadata_dir
    #speech_dates_file = options.dates_file
    #uscr_speech_dates_file = options.uscr_dates_file
    relevance_file = options.relevance_file
    keywords_segment_probs_file = options.keywords_segment_probs
    non_keywords_segment_probs_file = options.non_keywords_segment_probs
    uscr_transition = options.uscr_transition
    cfm_correction = options.cfm_correction
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
    #early_keywords_segments_file = os.path.join(keywords_dir, 'keyword_segments_' + str(early_first) + '-' + str(early_last) + '.jsonlist')
    #mid_keywords_segments_file = os.path.join(keywords_dir, 'keyword_segments_' + str(mid_first) + '-' + str(mid_last) + '.jsonlist')
    #modern_keywords_segments_file = os.path.join(keywords_dir, 'keyword_segments_' + str(modern_first) + '-' + str(modern_last) + '.jsonlist')
    #uscr_keywords_segments_file = os.path.join(keywords_dir, 'keyword_segments_uscr_' + str(uscr_first) + '-' + str(uscr_last) + '.jsonlist')

    #early_keywords_segments_file = os.path.join(keywords_dir, 'keyword_segments_' + str(early_first) + '-' + str(early_last) + '.jsonlist')
    #mid_keywords_segments_file = os.path.join(keywords_dir, 'keyword_segments_' + str(mid_first) + '-' + str(mid_last) + '.jsonlist')
    #modern_keywords_segments_file = os.path.join(keywords_dir, 'keyword_segments_' + str(modern_first) + '-' + str(modern_last) + '.jsonlist')
    #uscr_keywords_segments_file = os.path.join(keywords_dir, 'keyword_segments_uscr_' + str(uscr_first) + '-' + str(uscr_last) + '.jsonlist')

    #early_keywords_pred_file = os.path.join(keywords_dir, 'keyword_segments_' + str(early_first) + '-' + str(early_last) + '.jsonlist.early.binary_tone.tsv.tsv')
    #early_mid_keywords_pred_file = os.path.join(keywords_dir, 'keyword_segments_' + str(mid_first) + '-' + str(mid_last) + '.jsonlist.early.binary_tone.tsv.tsv')
    #modern_mid_keywords_pred_file = os.path.join(keywords_dir, 'keyword_segments_' + str(mid_first) + '-' + str(mid_last) + '.jsonlist.modern.binary_tone.tsv.tsv')
    #modern_keywords_pred_file = os.path.join(keywords_dir, 'keyword_segments_' + str(modern_first) + '-' + str(modern_last) + '.jsonlist.modern.binary_tone.tsv.tsv')
    #uscr_keywords_pred_file = os.path.join(keyword_pred_dir, 'keyword_segments_uscr_' + str(uscr_first) + '-' + str(uscr_last) + '.jsonlist.modern.binary_tone.tsv.tsv')

    #print("Loading keyword segments")
    #early_keyword_segments = load_segments(early_keywords_segments_file)
    #mid_keyword_segments = load_segments(mid_keywords_segments_file)
    #modern_keyword_segments = load_segments(modern_keywords_segments_file)
    #uscr_keyword_segments = load_segments(uscr_keywords_segments_file)


    print("Loading keyword data")
    # Read in files (broken up to avoid running out of memory)
    keywords_segments_file1 = os.path.join(keywords_dir, 'keyword_segments_all_43-88.jsonlist')
    keywords_segments_file2 = os.path.join(keywords_dir, 'keyword_segments_all_89-100.jsonlist')
    keywords_segments_file3 = os.path.join(keywords_dir, 'keyword_segments_all_101-108.jsonlist')
    keywords_segments_file4 = os.path.join(keywords_dir, 'keyword_segments_all_109-116.jsonlist')

    print("Loading keyword segments")
    keyword_segments1 = load_segments(keywords_segments_file1)
    keyword_segments2 = load_segments(keywords_segments_file2)
    keyword_segments3 = load_segments(keywords_segments_file3)
    keyword_segments4 = load_segments(keywords_segments_file4)

    keyword_segments1 = filter_segments(keyword_segments1, keywords_segment_probs)
    keyword_segments2 = filter_segments(keyword_segments2, keywords_segment_probs)
    keyword_segments3 = filter_segments(keyword_segments3, keywords_segment_probs)
    keyword_segments4 = filter_segments(keyword_segments4, keywords_segment_probs)

    print("Loading keyword preds")
    keyword_segment_probs1 = load_pred_probs(keywords_segments_file1 + '.binary_tone.tsv.tsv')
    keyword_segment_probs2 = load_pred_probs(keywords_segments_file2 + '.binary_tone.tsv.tsv')
    keyword_segment_probs3 = load_pred_probs(keywords_segments_file3 + '.binary_tone.tsv.tsv')
    keyword_segment_probs4 = load_pred_probs(keywords_segments_file4 + '.binary_tone.tsv.tsv')

    print("Counting speeches and getting max probs")
    #keyword_speech_probs = {}
    keyword_segment_probs = defaultdict(dict)
    keyword_label_counts = defaultdict(Counter)
    keyword_prob_sums = defaultdict(Counter)
    labeled_speech_dict = defaultdict(dict)
    keyword_label_counts, keyword_prob_sums, keywords_segment_probs, labeled_speech_dict = update_keyword_speech_probs(keyword_label_counts, keyword_prob_sums, keywords_segment_probs, keyword_segments1, keyword_segment_probs1, 43, 88, imm_speech_id_set, labeled_speech_dict)
    keyword_label_counts, keyword_prob_sums, keywords_segment_probs, labeled_speech_dict = update_keyword_speech_probs(keyword_label_counts, keyword_prob_sums, keywords_segment_probs, keyword_segments2, keyword_segment_probs2, 89, 100, imm_speech_id_set, labeled_speech_dict)
    keyword_label_counts, keyword_prob_sums, keywords_segment_probs, labeled_speech_dict = update_keyword_speech_probs(keyword_label_counts, keyword_prob_sums, keywords_segment_probs, keyword_segments3, keyword_segment_probs3, 101, 108, imm_speech_id_set, labeled_speech_dict)
    keyword_label_counts, keyword_prob_sums, keywords_segment_probs, labeled_speech_dict = update_keyword_speech_probs(keyword_label_counts, keyword_prob_sums, keywords_segment_probs, keyword_segments4, keyword_segment_probs4, 109, uscr_last, imm_speech_id_set, labeled_speech_dict)
    #keyword_label_counts, keyword_prob_sums, keywords_segment_probs, labeled_speech_dict = update_keyword_speech_probs(keyword_label_counts, keyword_prob_sums, keywords_segment_probs, keyword_segments4, keyword_segment_probs4, uscr_, uscr_transition-1, imm_speech_id_set, labeled_speech_dict)

    #uscr_keyword_speech_probs = {}
    #uscr_keyword_segment_probs = defaultdict(dict)
    #uscr_keyword_speech_probs, uscr_keyword_segment_probs = update_keyword_speech_probs(uscr_keyword_speech_probs, uscr_keyword_segment_probs, keyword_segments4, keyword_segment_probs4, uscr_transition, 116)
    #print(len(keyword_speech_probs), len(uscr_keyword_speech_probs))
    print(len(keyword_segment_probs))


    print("Loading metadata")
    id_to_party = {}
    files = sorted(glob(os.path.join(metadata_dir, 'metadata_*.json')))
    for infile in tqdm(files):
        with open(infile) as f:
            data = json.load(f)
        for key, vals in data.items():
            party = vals['party']
            id_to_party[key] = party

    print("Counting speeches and getting max probs")
    
    #keyword_label_counts, keyword_prob_sums, keywords_segment_probs, labeled_speech_dict = update_keyword_speech_probs(keyword_label_counts, keyword_prob_sums, keywords_segment_probs, early_keyword_segments, early_keyword_probs, early_first, early_last, imm_speech_id_set, labeled_speech_dict)
    #keyword_label_counts, keyword_prob_sums, keywords_segment_probs, labeled_speech_dict = update_keyword_speech_probs(keyword_label_counts, keyword_prob_sums, keywords_segment_probs, mid_keyword_segments, mid_keyword_probs, early_last+1, modern_first-1, imm_speech_id_set, labeled_speech_dict)
    #keyword_label_counts, keyword_prob_sums, keywords_segment_probs, labeled_speech_dict = update_keyword_speech_probs(keyword_label_counts, keyword_prob_sums, keywords_segment_probs, modern_keyword_segments, modern_keyword_probs, modern_first, uscr_transition-1, imm_speech_id_set, labeled_speech_dict)
    #keyword_label_counts, keyword_prob_sums, keywords_segment_probs, labeled_speech_dict = update_keyword_speech_probs(keyword_label_counts, keyword_prob_sums, keywords_segment_probs, uscr_keyword_segments, uscr_keyword_probs, uscr_transition, uscr_last, imm_speech_id_set, labeled_speech_dict, uscr=True)

    #keyword_prob_overall_sum = np.zeros(1)
    #for speech_id, probs in keyword_prob_sums.items():
    #    for p, v in probs.items():
    #        keyword_prob_overall_sum[p] += v
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
            segments = filter_segments(segments, non_keywords_segment_probs)
            pred_file = os.path.join(segments_pred_dir, 'pred.segments-' + str(congress).zfill(3) + '.tsv.tsv')
            pred_probs = load_pred_probs(pred_file)
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
                    for tone in range(len(TONES)):
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
    #df['keyword_neutral_count'] = [keyword_label_counts[i][1] for i in imm_speech_ids]
    df['keyword_pro_count'] = [keyword_label_counts[i][1] for i in imm_speech_ids]

    keyword_anti_probs = np.array([keyword_prob_sums[i][0] for i in imm_speech_ids])
    #keyword_netural_probs = np.array([keyword_prob_sums[i][1] for i in imm_speech_ids])
    keyword_pro_probs = np.array([keyword_prob_sums[i][1] for i in imm_speech_ids])

    df['keyword_anti_prob_sums'] = keyword_anti_probs
    #df['keyword_neutral_prob_sums'] = keyword_netural_probs
    df['keyword_pro_prob_sums'] = keyword_pro_probs

    # normalize with slight smoothing
    df['keyword_tone'] = (keyword_pro_probs) / (keyword_anti_probs + keyword_pro_probs)

    df['nonkeyword_anti_count'] = [all_label_counts[i][0] for i in imm_speech_ids]
    #df['nonkeyword_neutral_count'] = [all_label_counts[i][1] for i in imm_speech_ids]
    df['nonkeyword_pro_count'] = [all_label_counts[i][1] for i in imm_speech_ids]

    nonkeyword_anti_probs = np.array([all_prob_sums[i][0] for i in imm_speech_ids])
    #nonkeyword_netural_probs = np.array([all_prob_sums[i][1] for i in imm_speech_ids])
    nonkeyword_pro_probs = np.array([all_prob_sums[i][1] for i in imm_speech_ids])

    df['nonkeyword_anti_prob_sums'] = nonkeyword_anti_probs
    #df['nonkeyword_neutral_prob_sums'] = nonkeyword_netural_probs
    df['nonkeyword_pro_prob_sums'] = nonkeyword_pro_probs

    df['nonkeyword_tone'] = (nonkeyword_pro_probs) / (nonkeyword_anti_probs + nonkeyword_pro_probs)

    keyword_indicator = np.array(df['keyword'].values)
    overall_tone = keyword_indicator * np.array(df['keyword_tone'].values) + (1-keyword_indicator) * np.array(df['nonkeyword_tone'].values)
    overall_pro_sum = keyword_indicator * np.array(df['keyword_pro_prob_sums'].values) + (1-keyword_indicator) * np.array(df['nonkeyword_pro_prob_sums'].values)
    #overall_neutral_sum = keyword_indicator * np.array(df['keyword_neutral_prob_sums'].values) + (1-keyword_indicator) * np.array(df['nonkeyword_neutral_prob_sums'].values)
    overall_anti_sum = keyword_indicator * np.array(df['keyword_anti_prob_sums'].values) + (1-keyword_indicator) * np.array(df['nonkeyword_anti_prob_sums'].values)

    df['pro_prob_sum'] = overall_pro_sum
    #df['neutral_prob_sum'] = overall_neutral_sum
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
    df.columns = ['predicted', 'anti', 'pro']
    pred_labels = df['predicted'].values
    pred_scores_exp = np.exp(df[['anti', 'pro']].values)
    n_items, n_cats = pred_scores_exp.shape
    pred_probs = pred_scores_exp / pred_scores_exp.sum(1).reshape((n_items, 1))
    #yes_probs = pred_probs[:, 1]
    assert (pred_labels == np.argmax(pred_probs, axis=1)).all()
    return np.array(pred_probs)


def filter_segments(segments, keyword_segment_ids):
    outlines = []
    for line_i, line in enumerate(segments):
        segment_id = line['id']
        #parts = segment_id.split('_')
        #speech_id = parts[0]
        if segment_id in keyword_segment_ids:
            outlines.append(line)
    return outlines


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


def update_keyword_speech_probs(keyword_label_counts, keyword_prob_sums, keywords_segment_probs, segments, pred_probs, first_congress, last_congress, imm_speech_id_set, labeled_speech_dict):
    assert len(segments) == len(pred_probs)
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
                for tone in range(len(TONES)):
                    keyword_prob_sums[speech_id][tone] += pred_probs[line_i, tone]
                    keywords_segment_probs[segment_id][TONES[tone]] = pred_probs[line_i, tone]
                if speech_id in labeled_speech_dict:
                    labeled_speech_dict[speech_id]['keyword_probs'].append(list([float(v) for v in pred_probs[line_i]]))
                else:
                    labeled_speech_dict[speech_id]['keyword_probs'] = [list([float(v) for v in pred_probs[line_i]])]

    return keyword_label_counts, keyword_prob_sums, keywords_segment_probs, labeled_speech_dict



if __name__ == '__main__':
    main()
