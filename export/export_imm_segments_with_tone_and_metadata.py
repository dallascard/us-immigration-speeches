import os
import json
from optparse import OptionParser
from collections import defaultdict, Counter

import pandas as pd

# Combine the keyword and non-keyword segmets into one file / dataframe


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--keywords-dir', type=str, default='data/speeches/Congress/keyword_segments/',
                      help='Issue: default=%default')
    parser.add_option('--segments-dir', type=str, default='data/speeches/Congress/segments/',
                      help='Issue: default=%default')
    parser.add_option('--uscr-segments-dir', type=str, default='data/speeches/Congress/uscr_segments/',
                      help='Issue: default=%default')
    parser.add_option('--keyword-segments-file', type=str, default='data/speeches/Congress/keyword_segment_probs_selected_with_tone.json',
                      help='Issue: default=%default')
    parser.add_option('--non-keyword-segments-file', type=str, default='data/speeches/Congress/non_keyword_segment_probs_with_tone.json',
                      help='Issue: default=%default')
    parser.add_option('--uscr-transition', type=int, default=112,
                      help='The Congress at which to start using USCR instead of Gentzkow: default=%default')
    parser.add_option('--outdir', type=str, default='/u/scr/dcard/data/congress/immigration/predictions/',
                      help='Issue: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    keywords_dir = options.keywords_dir
    segments_dir = options.segments_dir
    uscr_segments_dir = options.uscr_segments_dir
    keywords_file = options.keyword_segments_file
    non_keywords_file = options.non_keyword_segments_file
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

    print("Loading keyword data")
    early_keywords_segments_file = os.path.join(keywords_dir, 'keyword_segments_' + str(early_first) + '-' + str(early_last) + '.jsonlist')
    mid_keywords_segments_file = os.path.join(keywords_dir, 'keyword_segments_' + str(mid_first) + '-' + str(mid_last) + '.jsonlist')
    modern_keywords_segments_file = os.path.join(keywords_dir, 'keyword_segments_' + str(modern_first) + '-' + str(modern_last) + '.jsonlist')
    uscr_keywords_segments_file = os.path.join(keywords_dir, 'keyword_segments_uscr_' + str(uscr_first) + '-' + str(uscr_last) + '.jsonlist')

    with open(keywords_file) as f:
        keywords_probs = json.load(f)

    with open(non_keywords_file) as f:
        non_keywords_probs = json.load(f)

    keyword_outlines = []
    print("Loading keyword segments")
    for seg_file in [early_keywords_segments_file, mid_keywords_segments_file, modern_keywords_segments_file, uscr_keywords_segments_file]:
        print(seg_file)
        keyword_segments = load_segments(seg_file)
        for segment in keyword_segments:
            segment_id = segment['id']
            if segment_id in keywords_probs:
                line = keywords_probs[segment_id]
                line['segment_id'] = segment_id
                line['text'] = segment['text']
                keyword_outlines.append(line)

    found_segments = set([line['segment_id'] for line in keyword_outlines])
    print(len(found_segments), len(keywords_probs))
    print(len(found_segments - set(keywords_probs.keys())), len(set(keywords_probs.keys()) - found_segments))

    # then to non-keyword segments
    non_keyword_outlines = []
    for congress in range(early_first, uscr_last+1):
        if congress < uscr_transition:
            segments_file = os.path.join(segments_dir, 'segments-' + str(congress).zfill(3) + '.jsonlist')
        else:
            segments_file = os.path.join(uscr_segments_dir, 'segments-' + str(congress).zfill(3) + '.jsonlist')
        print(segments_file)
        segments = load_segments(segments_file)
        for segment in segments:
            segment_id = segment['id']
            if segment_id in non_keywords_probs:
                line = non_keywords_probs[segment_id]
                line['segment_id'] = segment_id
                line['text'] = segment['text']
                non_keyword_outlines.append(line)

    found_segments = set([line['segment_id'] for line in non_keyword_outlines])
    print(len(found_segments), len(non_keywords_probs))
    print(len(found_segments - set(non_keywords_probs.keys())), len(set(non_keywords_probs.keys()) - found_segments))
    for i in sorted(set(non_keywords_probs.keys()) - found_segments)[:100]:
        print(i)

    outlines = keyword_outlines + non_keyword_outlines
    outfile = os.path.join(outdir, 'imm_segments_with_tone_and_metadata.jsonlist')
    with open(outfile, 'w') as f:
        for line in outlines:
            f.write(json.dumps(line) + '\n')

    df = convert_to_dataframe(outlines)
    outfile = os.path.join(outdir, 'imm_segments_with_tone_and_metadata.tsv')
    df.to_csv(outfile, sep='\t')


def load_segments(infile):
    segments = []
    with open(infile) as f:
        for line in f:
            segments.append(json.loads(line))
    return segments


def extract_speech_id_from_segment_id(segment_id):
    speech_id = '_'.join(segment_id.split('_')[:-1])
    return speech_id


def convert_to_dataframe(outlines):
    df = pd.DataFrame()
    df['segment_id'] = [line['segment_id'] for line in outlines]
    df['speech_id'] = [line['speech_id'] for line in outlines]
    df['segment'] = [line['segment'] for line in outlines]
    df['congress'] = [line['congress'] for line in outlines]
    df['date'] = [line['date'] for line in outlines]
    df['year'] = [int(str(line['date'])[:4]) for line in outlines]
    df['chamber'] = [line['chamber'] for line in outlines]
    df['party'] = [line['party'] for line in outlines]
    df['state'] = [line['state'] for line in outlines]
    df['speaker'] = [line['speaker'] for line in outlines]
    df['speaker_id'] = [line['speaker_id'] for line in outlines]
    df['uscr'] = [line['uscr'] for line in outlines]
    df['keyword'] = [line['keyword'] for line in outlines]
    df['imm_score'] = [line['prob'] for line in outlines]
    df['pro_score'] = [line['pro'] for line in outlines]
    df['neutral_score'] = [line['neutral'] for line in outlines]
    df['anti_score'] = [line['anti'] for line in outlines]
    df['text'] = [line['text'] for line in outlines]
    return df


if __name__ == '__main__':
    main()
