import os
import json
from optparse import OptionParser
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

from uscr.export_speeches import date_to_congress as uscr_date_to_congress


# Collect all predictions on keyword and non-keyword segments and combine into predictions for each speech
# Only keep non-keyword predictions for speeches that weren't selected via keywords
# and choose here where to switch over to USCR data


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--segments-dir', type=str, default='data/speeches/Congress/segments/',
                      help='Directory with raw segments: default=%default')
    parser.add_option('--uscr-segments-dir', type=str, default='data/speeches/Congress/uscr_segments/',
                      help='Directory with raw segments: default=%default')
    parser.add_option('--keywords-dir', type=str, default='data/speeches/Congresscongress/keyword_segments/',
                      help='Directory with model predictions: default=%default')
    parser.add_option('--early-model-dir', type=str, default='data/speeches/Congress/data/congress_early/relevance/splits/label-weights/all/all__s1040749045_lr2e-05_msl512/',
                      help='Directory with model predictions: default=%default')
    parser.add_option('--modern-model-dir', type=str, default='data/speeches/Congress/data/congress_modern/relevance/label-weights/all/all__s2405527281_lr2e-05_msl512/',
                      help='Directory with model predictions: default=%default')
    parser.add_option('--metadata-dir', type=str, default='data/speeches/Congress/metadata/',
                      help='Directory with model predictions: default=%default')
    parser.add_option('--dates-file', type=str, default='/u/scr/nlp/data/congress/speech_dates.json',
                      help='File with speech dates: default=%default')
    parser.add_option('--uscr-dates-file', type=str, default='/u/scr/nlp/data/congress/uscr_speech_dates.json',
                      help='File with speech dates: default=%default')
    parser.add_option('--uscr-transition', type=int, default=112,
                      help='The Congress at which to start using USCR instead of Gentzkow: default=%default')
    parser.add_option('--outdir', type=str, default='/u/scr/dcard/data/congress/',
                      help='Outdir: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    segments_dir = options.segments_dir
    uscr_segments_dir = options.uscr_segments_dir
    keywords_dir = options.keywords_dir
    early_model_dir = options.early_model_dir
    modern_model_dir = options.modern_model_dir
    metadata_dir = options.metadata_dir
    speech_dates_file = options.dates_file
    uscr_speech_dates_file = options.uscr_dates_file
    uscr_transition = options.uscr_transition
    outdir = options.outdir

    early_first = 43
    early_last = 73
    mid_first = 70
    mid_last = 88
    modern_first = 85
    modern_last = 114
    uscr_first = 104
    uscr_last = 116

    print("Loading speech dates")
    with open(speech_dates_file) as f:
        speech_dates = json.load(f)
    with open(uscr_speech_dates_file) as f:
        uscr_speech_dates = json.load(f)
    all_speech_dates = {}
    all_speech_dates.update(speech_dates)
    all_speech_dates.update(uscr_speech_dates)

    speech_chamber = {}
    speech_party = {}
    speech_state = {}
    speech_speaker = {}
    speech_speaker_id = {}
    speech_congress = {}
    print("Loading metadata")
    for congress in range(early_first, uscr_last+1):
        # load metadta for both hein and uscr corpora (speech_ids don't overlap)
        if congress <= modern_last:
            metadata_file = os.path.join(metadata_dir, 'metadata_' + str(congress).zfill(3) + '.json')
            with open(metadata_file) as f:
                metadata = json.load(f)
            for speech_id, data in metadata.items():
                speech_chamber[speech_id] = data['chamber']
                speech_party[speech_id] = data['party']
                speech_state[speech_id] = data['state']
                speech_speaker[speech_id] = data['speaker']
                speech_speaker_id[speech_id] = data['speaker_id']
                speech_congress[speech_id] = congress
        if congress >= uscr_first:
            metadata_file = os.path.join(metadata_dir, 'uscr_metadata_' + str(congress).zfill(3) + '.json')
            with open(metadata_file) as f:
                metadata = json.load(f)
            for speech_id, data in metadata.items():
                speech_chamber[speech_id] = data['chamber']
                speech_party[speech_id] = data['party']
                speech_state[speech_id] = data['state']
                speech_speaker[speech_id] = data['speaker']
                speech_speaker_id[speech_id] = data['speaker_id']
                speech_congress[speech_id] = congress

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

    mid_keyword_probs = interpolate_mid_probs(mid_keyword_segments, early_mid_keyword_probs, modern_mid_keyword_probs, early_last, modern_first)

    print("Counting speeches and getting max probs")
    keyword_speech_probs = {}
    keyword_segment_probs = defaultdict(dict)
    keyword_speech_probs, keyword_segment_probs = update_keyword_speech_probs(keyword_speech_probs, keyword_segment_probs, early_keyword_segments, early_keyword_probs, early_first, early_last)
    keyword_speech_probs, keyword_segment_probs,  = update_keyword_speech_probs(keyword_speech_probs, keyword_segment_probs, mid_keyword_segments, mid_keyword_probs, early_last+1, modern_first-1)
    keyword_speech_probs, keyword_segment_probs = update_keyword_speech_probs(keyword_speech_probs, keyword_segment_probs, modern_keyword_segments, modern_keyword_probs, modern_first, uscr_transition-1)
    uscr_keyword_speech_probs = {}
    uscr_keyword_segment_probs = defaultdict(dict)
    uscr_keyword_speech_probs, uscr_keyword_segment_probs = update_keyword_speech_probs(uscr_keyword_speech_probs, uscr_keyword_segment_probs, uscr_keyword_segments, uscr_keyword_probs, uscr_transition, uscr_last, uscr=True)
    print(len(keyword_speech_probs), len(uscr_keyword_speech_probs))
    print(len(keyword_segment_probs))

    imm_speech_ids = sorted(keyword_speech_probs)
    imm_probs_list = [keyword_speech_probs[i] for i in imm_speech_ids]
    imm_speech_dates = [speech_dates[i] for i in imm_speech_ids]
    imm_keyword_indicators = [1] * len(imm_speech_ids)
    imm_keyword_chamber = [speech_chamber[i] for i in imm_speech_ids]
    imm_keyword_party = [speech_party[i] for i in imm_speech_ids]
    imm_keyword_state = [speech_state[i] for i in imm_speech_ids]
    imm_keyword_speaker = [speech_speaker[i] for i in imm_speech_ids]
    imm_keyword_speaker_id = [speech_speaker_id[i] for i in imm_speech_ids]
    imm_keyword_uscr_indicators = [0] * len(imm_speech_ids)
    imm_keyword_congresses = [speech_congress[i] for i in imm_speech_ids]

    uscr_keyword_speech_ids = sorted(uscr_keyword_speech_probs)
    imm_speech_ids.extend(uscr_keyword_speech_ids)
    imm_probs_list.extend([uscr_keyword_speech_probs[i] for i in uscr_keyword_speech_ids])
    imm_speech_dates.extend([uscr_speech_dates[i] for i in uscr_keyword_speech_ids])
    imm_keyword_indicators.extend([1] * len(uscr_keyword_speech_ids))
    imm_keyword_chamber.extend([speech_chamber[i] for i in uscr_keyword_speech_ids])
    imm_keyword_party.extend([speech_party[i] for i in uscr_keyword_speech_ids])
    imm_keyword_state.extend([speech_state[i] for i in uscr_keyword_speech_ids])
    imm_keyword_speaker.extend([speech_speaker[i] for i in uscr_keyword_speech_ids])
    imm_keyword_speaker_id.extend([speech_speaker_id[i] for i in uscr_keyword_speech_ids])
    imm_keyword_uscr_indicators.extend([1] * len(uscr_keyword_speech_ids))
    imm_keyword_congresses.extend([speech_congress[i] for i in uscr_keyword_speech_ids])

    keyword_speech_ids = set(imm_speech_ids).union(uscr_keyword_speech_ids)

    keyword_segment_ids = sorted(keyword_segment_probs)
    for segment_id in keyword_segment_ids:
        parts = segment_id.split('_')
        speech_id = '_'.join(parts[:-1])
        keyword_segment_probs[segment_id]['date'] = all_speech_dates[speech_id]
        keyword_segment_probs[segment_id]['chamber'] = speech_chamber[speech_id]
        keyword_segment_probs[segment_id]['party'] = speech_party[speech_id]
        keyword_segment_probs[segment_id]['state'] = speech_state[speech_id]
        keyword_segment_probs[segment_id]['speaker'] = speech_speaker[speech_id]
        keyword_segment_probs[segment_id]['speaker_id'] = speech_speaker_id[speech_id]
        keyword_segment_probs[segment_id]['uscr'] = 0
        keyword_segment_probs[segment_id]['congress'] = speech_congress[speech_id]

    # add in the uscr keyword segments
    uscr_keyword_segment_ids = sorted(uscr_keyword_segment_probs)
    for segment_id in uscr_keyword_segment_ids:
        parts = segment_id.split('_')
        speech_id = '_'.join(parts[:-1])
        keyword_segment_probs[segment_id]['speech_id'] = uscr_keyword_segment_probs[segment_id]['speech_id']
        keyword_segment_probs[segment_id]['segment'] = uscr_keyword_segment_probs[segment_id]['segment']
        keyword_segment_probs[segment_id]['prob'] = uscr_keyword_segment_probs[segment_id]['prob']
        keyword_segment_probs[segment_id]['keyword'] = 1
        keyword_segment_probs[segment_id]['date'] = all_speech_dates[speech_id]
        keyword_segment_probs[segment_id]['chamber'] = speech_chamber[speech_id]
        keyword_segment_probs[segment_id]['party'] = speech_party[speech_id]
        keyword_segment_probs[segment_id]['state'] = speech_state[speech_id]
        keyword_segment_probs[segment_id]['speaker'] = speech_speaker[speech_id]
        keyword_segment_probs[segment_id]['speaker_id'] = speech_speaker_id[speech_id]
        keyword_segment_probs[segment_id]['uscr'] = 1
        keyword_segment_probs[segment_id]['congress'] = speech_congress[speech_id]

    # create a dict for non-keyword segment probs
    other_segment_probs = defaultdict(dict)

    # count the number of imm keyword speeches per day
    imm_speeches_per_day = Counter()
    for speech_id, date in speech_dates.items():
        if speech_id in keyword_speech_probs:
            imm_speeches_per_day[date] += 1
    uscr_imm_speeches_per_day = Counter()
    for speech_id, date in uscr_speech_dates.items():
        if speech_id in uscr_keyword_speech_probs:
            uscr_imm_speeches_per_day[date] += 1

    outfile = os.path.join(outdir, 'imm_speeches_per_day.json')
    with open(outfile, 'w') as f:
        json.dump(imm_speeches_per_day, f, indent=2)

    outfile = os.path.join(outdir, 'uscr_imm_speeches_per_day.json')
    with open(outfile, 'w') as f:
        json.dump(uscr_imm_speeches_per_day, f, indent=2)

    # add non-keyword speeches for each congress
    for uscr in [0, 1]:
        # don't overlap the Genzkow and USCR data
        if uscr:
            start = uscr_transition
            end = uscr_last
        else:
            start = early_first
            end = uscr_transition-1
        for congress in range(start, end+1):
            temp_segment_probs = defaultdict(dict)
            if uscr:
                segments_file = os.path.join(uscr_segments_dir, 'segments-' + str(congress).zfill(3) + '.jsonlist')
            else:
                segments_file = os.path.join(segments_dir, 'segments-' + str(congress).zfill(3) + '.jsonlist')
            segments = load_segments(segments_file)
            if uscr:
                pred_file = os.path.join(modern_model_dir, 'pred.segments.uscr.new-' + str(congress).zfill(3) + '.tsv.tsv')
                yes_probs = load_pred_probs(pred_file)
            elif congress <= early_last:
                pred_file = os.path.join(early_model_dir, 'pred.segments-' + str(congress).zfill(3) + '.tsv.tsv')
                yes_probs = load_pred_probs(pred_file)
            elif congress >= modern_first:
                pred_file = os.path.join(modern_model_dir, 'pred.segments-' + str(congress).zfill(3) + '.tsv.tsv')
                yes_probs = load_pred_probs(pred_file)
            else:
                early_pred_file = os.path.join(early_model_dir, 'pred.segments-' + str(congress).zfill(3) + '.tsv.tsv')
                early_yes_probs = load_pred_probs(early_pred_file)
                modern_pred_file = os.path.join(modern_model_dir, 'pred.segments-' + str(congress).zfill(3) + '.tsv.tsv')
                modern_yes_probs = load_pred_probs(modern_pred_file)
                modern_weight = min(max((congress - early_last) / (modern_first - early_last), 0.), 1.)
                yes_probs = (1. - modern_weight) * early_yes_probs + modern_weight * modern_yes_probs
            assert len(yes_probs) == len(segments)

            # get the max segment probability associated with each speech
            unfilitered_speech_probs = defaultdict(float)
            for line_i, line in enumerate(segments):
                segment_id = line['id']
                parts = segment_id.split('_')
                if uscr:
                    speech_id = '_'.join(parts[:-1])
                else:
                    speech_id = parts[0]

                # exclude the cases where I've included the table header by mistake
                if speech_id != 'speech':
                    unfilitered_speech_probs[speech_id] = max(unfilitered_speech_probs[speech_id], yes_probs[line_i])
                    if yes_probs[line_i] > 0.5:
                        temp_segment_probs[speech_id][segment_id] = yes_probs[line_i]

            new_speech_probs = defaultdict(float)
            count = 0
            # choose segments to add for each speech with any prob >= 0.5
            for speech_id, prob in unfilitered_speech_probs.items():
                # only keep predictions for speeches that we haven't already selected via keywords
                if speech_id not in keyword_speech_ids:
                    if uscr:
                        speech_date = uscr_speech_dates[speech_id]
                        imm_speeches_on_date = uscr_imm_speeches_per_day[speech_date]
                    else:
                        speech_date = speech_dates[speech_id]
                        imm_speeches_on_date = imm_speeches_per_day[speech_date]

                    # require a prediction of immigration
                    if prob > 0.5:
                        count += 1
                        # in combination with a day discussing immigration
                        if (prob + imm_speeches_on_date / 100.) > 1.0:
                            if uscr:
                                if speech_id not in uscr_keyword_speech_probs:
                                    new_speech_probs[speech_id] = prob
                            else:
                                if speech_id not in keyword_speech_probs:
                                    new_speech_probs[speech_id] = prob

                            # for those speeches which are identified as being about immigration
                            # get the specific segments with probability > 0.5
                            for segment_id, prob in temp_segment_probs[speech_id].items():
                                # split off last part of identifier and drop first character (b)
                                batch = segment_id.split('_')[-1]
                                assert batch[0] == 'b'
                                num = int(batch[1:])
                                # save all selected (non-keyword) segments that have not been filtered out
                                other_segment_probs[segment_id] = {'speech_id': speech_id, 'segment': num, 'prob': prob, 'keyword': 0, 'uscr': uscr, 'congress': congress}

            print(uscr, congress, count, len(new_speech_probs))

            # add the newly selected (non-keyword) speeches to the earlier data
            new_speech_ids = sorted(new_speech_probs)
            imm_speech_ids.extend(new_speech_ids)
            imm_probs_list.extend([new_speech_probs[i] for i in new_speech_ids])
            imm_speech_dates.extend([all_speech_dates[i] for i in new_speech_ids])
            imm_keyword_indicators.extend([0] * len(new_speech_ids))
            imm_keyword_chamber.extend([speech_chamber[i] for i in new_speech_ids])
            imm_keyword_party.extend([speech_party[i] for i in new_speech_ids])
            imm_keyword_state.extend([speech_state[i] for i in new_speech_ids])
            imm_keyword_speaker.extend([speech_speaker[i] for i in new_speech_ids])
            imm_keyword_speaker_id.extend([speech_speaker_id[i] for i in new_speech_ids])
            if uscr:
                imm_keyword_uscr_indicators.extend([1] * len(new_speech_ids))
            else:
                imm_keyword_uscr_indicators.extend([0] * len(new_speech_ids))
            imm_keyword_congresses.extend([congress] * len(new_speech_ids))

    order = np.argsort(imm_speech_dates)
    df = pd.DataFrame()
    df['speech_id'] = [imm_speech_ids[i] for i in order]
    df['congress'] = [imm_keyword_congresses[i] for i in order]
    df['date'] = sorted(imm_speech_dates)
    df['imm_prob'] = [imm_probs_list[i] for i in order]
    df['keyword'] = [imm_keyword_indicators[i] for i in order]
    df['chamber'] = [imm_keyword_chamber[i] for i in order]
    df['party'] = [imm_keyword_party[i] for i in order]
    df['state'] = [imm_keyword_state[i] for i in order]
    df['speaker'] = [imm_keyword_speaker[i] for i in order]
    df['speaker_id'] = [imm_keyword_speaker_id[i] for i in order]
    df['uscr'] = [imm_keyword_uscr_indicators[i] for i in order]

    non_keyword_segment_ids = sorted(other_segment_probs)
    for segment_id in non_keyword_segment_ids:
        parts = segment_id.split('_')
        speech_id = '_'.join(parts[:-1])
        other_segment_probs[segment_id]['date'] = all_speech_dates[speech_id]
        other_segment_probs[segment_id]['chamber'] = speech_chamber[speech_id]
        other_segment_probs[segment_id]['party'] = speech_party[speech_id]
        other_segment_probs[segment_id]['state'] = speech_state[speech_id]
        other_segment_probs[segment_id]['speaker'] = speech_speaker[speech_id]
        other_segment_probs[segment_id]['speaker_id'] = speech_speaker_id[speech_id]

    print(df.shape)
    outfile = os.path.join(outdir, 'imm_speech_ids_all.tsv')
    df.to_csv(outfile, sep='\t')

    outfile = os.path.join(outdir, 'keyword_segment_probs_selected.json')
    with open(outfile, 'w') as f:
        json.dump(keyword_segment_probs, f, indent=2)

    outfile = os.path.join(outdir, 'non_keyword_segment_probs.json')
    with open(outfile, 'w') as f:
        json.dump(other_segment_probs, f, indent=2)


def load_segments(infile):
    segments = []
    with open(infile) as f:
        for line in f:
            segments.append(json.loads(line))
    return segments


def load_pred_probs(infile):
    df = pd.read_csv(infile, header=0, index_col=None, sep='\t')
    df.columns = ['predicted', 'no', 'yes']
    pred_labels = df['predicted'].values
    pred_scores_exp = np.exp(df[['no', 'yes']].values)
    n_items, n_cats = pred_scores_exp.shape
    pred_probs = pred_scores_exp / pred_scores_exp.sum(1).reshape((n_items, 1))
    yes_probs = pred_probs[:, 1]
    assert (pred_labels == (yes_probs >= 0.5)).all()
    return np.array(yes_probs)


def interpolate_mid_probs(segments, early_probs, modern_probs, early_last, modern_first):
    interpolated_probs = []
    for line_i, line in enumerate(segments):
        segment_id = line['id']
        parts = segment_id.split('_')
        speech_id = parts[0]
        congress = int(speech_id[:2])
        modern_weight = min(max((congress - early_last) / (modern_first - early_last), 0.), 1.)
        interpolated_probs.append((1. - modern_weight) * early_probs[line_i] + modern_weight * modern_probs[line_i])
    return interpolated_probs


def update_keyword_speech_probs(keyword_speech_probs, keyword_segment_probs, segments, yes_probs, first_congress, last_congress, uscr=False):

    temp_keyword_segment_probs = defaultdict(dict)

    assert len(segments) == len(yes_probs)
    for line_i, line in enumerate(segments):
        segment_id = line['id']
        parts = segment_id.split('_')
        # get the centered sentence number
        segment_num = int(parts[-1])
        if uscr:
            speech_id = '_'.join(parts[:-1])
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
            pred_prob = yes_probs[line_i]
            if pred_prob >= 0.5:
                if speech_id in keyword_speech_probs:
                    keyword_speech_probs[speech_id] = max(keyword_speech_probs[speech_id], pred_prob)
                else:
                    keyword_speech_probs[speech_id] = pred_prob
                # store all keyword probs >= 0.5
                temp_keyword_segment_probs[speech_id][segment_num] = pred_prob

    # prune the keyword segment probs (greedily) to avoid excessive overlap
    for speech_id, segment_probs in temp_keyword_segment_probs.items():
        # if there is only one segment, just take it
        if len(segment_probs) == 1:
            for num, prob in segment_probs.items():
                keyword_segment_probs[speech_id + '_' + str(num)] = {'speech_id': speech_id, 'segment': num, 'prob': prob, 'keyword': 1}
        else:
            # get the (overlapping) segments and their probs
            segment_nums = sorted(segment_probs)
            remaining_set = set([int(i) for i in segment_nums])
            probs = [segment_probs[s] for s in segment_nums]
            # sort them by probability
            order = np.argsort(probs)[::-1]
            # create a block list
            for i in order:
                # get the segment number
                num = int(segment_nums[i])
                # check if we haven't excluded this segment yet
                if num in remaining_set:
                    prob = probs[i]
                    # add it's segment prob to the output
                    keyword_segment_probs[speech_id + '_' + str(num)] = {'speech_id': speech_id, 'segment': num, 'prob': prob, 'keyword': 1}
                    # block the segments around it
                    remaining_set = remaining_set - set(range(num-5, num+6))

    return keyword_speech_probs, keyword_segment_probs


if __name__ == '__main__':
    main()
