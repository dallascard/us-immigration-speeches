import os
import json
import datetime as dt
from glob import glob
from optparse import OptionParser
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sms
import matplotlib.pyplot as plt

import smart_open
smart_open.open = smart_open.smart_open

from analysis.frame_terms import get_frame_replacements
from time_periods.common import year_to_congress, congress_to_year


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--tone-file', type=str, default='data/speeche/Congress/imm_speech_ids_with_tone.tsv',
                      help='.tsv file with immigration segments and tones: default=%default')
    parser.add_option('--procedural-file', type=str, default='data/speeche/Congress/procedural_speech_ids.txt',
                      help='File with list of procdeural speeches: default=%default')
    parser.add_option('--imm-parsed-file', type=str, default='data/speeche/Congress/imm_mention_sents_parsed.jsonlist',
                      help='File with parsed sentences containing immigrant mentions (from analysis.identify_immigrant_mentions.py): default=%default')
    parser.add_option('--imm-group-file', type=str, default='data/speeche/Congress/tagged_counts/imm_mention_sent_indices_by_group.json',
                      help='Sent indices of mention sentences with group mentions (from analysis.identify_group_mentions.py): default=%default')
    parser.add_option('--hein-parsed-dir', type=str, default='data/speeche/Congress/hein-bound_parsed/',
                      help='Directory with parsed hein bound speeches: default=%default')
    parser.add_option('--uscr-parsed-dir', type=str, default='data/speeche/Congress/uscr_parsed/',
                      help='Directory with parsed USCR speeches: default=%default')
    parser.add_option('--metadata-dir', type=str, default='data/speeche/Congress/metadata/',
                      help='Metadata directory: default=%default')
    parser.add_option('--outdir', type=str, default='data/speeche/Congress/tagged_counts/',
                      help='Output dir: default=%default')
    parser.add_option('--first-congress', type=int, default=46,
                      help='First congress: default=%default')
    parser.add_option('--last-congress', type=int, default=116,
                      help='Last congress: default=%default')
    parser.add_option('--uscr-transition', type=int, default=112,
                      help='Cognress at which to start using USCR dat: default=%default')

    (options, args) = parser.parse_args()

    tone_file = options.tone_file
    procedural_file = options.procedural_file
    imm_parsed_file = options.imm_parsed_file
    imm_group_file = options.imm_group_file
    hein_parsed_dir = options.hein_parsed_dir
    uscr_parsed_dir = options.uscr_parsed_dir
    metadata_dir = options.metadata_dir
    outdir = options.outdir
    first_congress = options.first_congress
    last_congress = options.last_congress
    uscr_transition = options.uscr_transition

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    congress_range = list(range(first_congress, last_congress+1))
    years = [congress_to_year(c) for c in congress_range]
    print(years[0], years[-1])

    print("Loading speech data")
    df_new = pd.read_csv(tone_file, header=0, index_col=0, sep='\t')
    df_new.head()

    # Load speeches to exclude
    print("Loading procedural speech ids")
    with open(procedural_file) as f:
        to_exclude = f.readlines()
    to_exclude = set([s.strip() for s in to_exclude])
    print(len(to_exclude))

    print("Loading parsed sentences with mentions of immigrants")
    with open(imm_parsed_file) as f:
        imm_sent_lines = f.readlines()
    imm_sent_lines = [json.loads(line) for line in imm_sent_lines]
    print(len(imm_sent_lines))

    with open(imm_group_file) as f:
        imm_sent_indices_by_group = json.load(f)
    groups = sorted(imm_sent_indices_by_group)
    for group in groups:
        imm_sent_indices_by_group[group] = set(imm_sent_indices_by_group[group])

    imm_speech_id_list = [str(i) for i in df_new['speech_id'].values]
    imm_speech_ids = set(imm_speech_id_list)
    party_by_speech_id = dict(zip(imm_speech_id_list, df_new['party'].values))

    frame_replacements = get_frame_replacements()

    # Count tokens in sentenecs mentioning immigrants
    imm_sent_token_counts_by_congress = defaultdict(Counter)
    imm_sent_token_counts_by_congress_by_party = {p: defaultdict(Counter) for p in ['D', 'R']}
    imm_sent_token_counts_by_congress_by_group = {p: defaultdict(Counter) for p in ['Chinese', 'Mexican', 'European']}

    # Collect the tokens from the immigrant mention sentences
    query_bigram_counter = Counter()
    lines_by_group = Counter()
    for line in tqdm(imm_sent_lines):
        speech_id = line['id']
        sent_index = line['sent_index']
        if speech_id not in to_exclude:
            if speech_id.startswith('C'):
                year = int(speech_id.split('-')[1])
                congress = year_to_congress(year)
            elif speech_id.startswith('1'):
                congress = int(speech_id[:3])
            else:
                congress = int(speech_id[:2])
            lemmas = [lemma.lower() for lemma in line['lemmas']]
            tags = line['tags']

            tagged_lemmas = [lemma + ' (' + tags[i][0].lower() + ')' for i, lemma in enumerate(lemmas)]
            # look for ones we're missing because of different parsing
            additional = []
            if len(tagged_lemmas) > 1:
                for l_i, lemma in enumerate(tagged_lemmas[:-1]):
                    if lemma + ' ' + tagged_lemmas[l_i+1] in frame_replacements:
                        additional.append(frame_replacements[lemma + ' ' + tagged_lemmas[l_i+1]])
                        query_bigram_counter[tagged_lemmas[l_i] + ' ' + tagged_lemmas[l_i+1]] += 1
            tagged_lemmas.extend(additional)

            imm_sent_token_counts_by_congress[congress].update(tagged_lemmas)
            party = party_by_speech_id[speech_id]
            if party == 'D' or party == 'R':
                imm_sent_token_counts_by_congress_by_party[party][congress].update(tagged_lemmas)

            # check if this is a group mention sent (based on other script)
            sent_id = str(speech_id) + '_' + str(sent_index)
            if sent_id in imm_sent_indices_by_group['Mexican']:
                imm_sent_token_counts_by_congress_by_group['Mexican'][congress].update(tagged_lemmas)
            if sent_id in imm_sent_indices_by_group['European']:
                imm_sent_token_counts_by_congress_by_group['European'][congress].update(tagged_lemmas)
            if sent_id in imm_sent_indices_by_group['Chinese']:
                imm_sent_token_counts_by_congress_by_group['Chinese'][congress].update(tagged_lemmas)
            if sent_id in imm_sent_indices_by_group['Hispanic']:
                imm_sent_token_counts_by_congress_by_group['Hispanic'][congress].update(tagged_lemmas)

    # Save these token counts
    with open(os.path.join(outdir, 'imm_sent_token_counts_by_congress.json'), 'w') as f:
        json.dump(imm_sent_token_counts_by_congress, f, indent=2)

    with open(os.path.join(outdir, 'imm_sent_token_counts_by_congress_by_party.json'), 'w') as f:
        json.dump(imm_sent_token_counts_by_congress_by_party, f, indent=2)

    with open(os.path.join(outdir, 'imm_sent_token_counts_by_congress_by_group.json'), 'w') as f:
        json.dump(imm_sent_token_counts_by_congress_by_group, f, indent=2)

    # Count tokens in all speeches
    token_counts_by_congress = defaultdict(Counter)
    token_counts_by_congress_by_party = {p: defaultdict(Counter) for p in ['D', 'R']}
    imm_token_counts_by_congress = defaultdict(Counter)
    imm_token_counts_by_congress_by_party = {p: defaultdict(Counter) for p in ['D', 'R']}

    for congress in range(first_congress, last_congress+1):
        print(congress)
        if congress < uscr_transition:
            # Note that I saved the parsed files as .txt by mistake; they are actually .jsonlist
            infile = os.path.join(hein_parsed_dir, 'speeches_' + str(congress).zfill(3) + '.txt')
            metadata_file = os.path.join(metadata_dir, 'metadata_' + str(congress).zfill(3) + '.json')
        else:
            infile = os.path.join(uscr_parsed_dir, 'speeches_' + str(congress).zfill(3) + '.jsonlist')
            metadata_file = os.path.join(metadata_dir, 'uscr_metadata_' + str(congress).zfill(3) + '.json')
        with open(metadata_file) as f:
            metadata = json.load(f)

        with open(infile) as f:
            for line in tqdm(f):
                line = json.loads(line)
                speech_id = str(line['id'])
                if speech_id in to_exclude:
                    pass
                elif speech_id != 'speech_id':
                    party = metadata[speech_id]['party']
                    lemma_counter = Counter()
                    tagged_lemma_counter = Counter()
                    tags = line['tags']
                    for sent_i, sent in enumerate(line['lemmas']):
                        tokens = [t.lower() for t in sent]
                        lemma_counter.update(tokens)
                        tagged_lemmas = [token.lower() + ' (' + tags[sent_i][i][0].lower() + ')' for i, token in enumerate(sent)]
                        # look for ones we're missing because of different parsing
                        additional = []
                        if len(tokens) > 1:
                            for l_i, lemma in enumerate(tagged_lemmas[:-1]):
                                if lemma + ' ' + tagged_lemmas[l_i+1] in frame_replacements:
                                    additional.append(frame_replacements[lemma + ' ' + tagged_lemmas[l_i+1]])
                                    query_bigram_counter[tagged_lemmas[l_i] + ' ' + tagged_lemmas[l_i+1]] += 1
                        tagged_lemmas.extend(additional)
                        tagged_lemma_counter.update(tagged_lemmas)

                    token_counts_by_congress[congress].update(tagged_lemma_counter)

                    if party == 'D' or party == 'R':
                        token_counts_by_congress_by_party[party][congress].update(tagged_lemma_counter)
                    if speech_id in imm_speech_ids:
                        #tone = tones_by_speech_id[speech_id]
                        imm_token_counts_by_congress[congress].update(tagged_lemma_counter)
                        #imm_token_counts_by_congress_by_tone[tone][congress].update(tagged_lemma_counter)
                        if party == 'D' or party == 'R':
                            imm_token_counts_by_congress_by_party[party][congress].update(tagged_lemma_counter)

    with open(os.path.join(outdir, 'query_bigrams.json'), 'w') as f:
        json.dump(query_bigram_counter.most_common(), f, indent=2, sort_keys=False)

    with open(os.path.join(outdir, 'token_counts_by_congress.json'), 'w') as f:
        json.dump(token_counts_by_congress, f, indent=2)

    with open(os.path.join(outdir, 'token_counts_by_congress_by_party.json'), 'w') as f:
        json.dump(token_counts_by_congress_by_party, f, indent=2)

    with open(os.path.join(outdir, 'imm_token_counts_by_congress.json'), 'w') as f:
        json.dump(imm_token_counts_by_congress, f, indent=2)

    with open(os.path.join(outdir, 'imm_token_counts_by_congress_by_party.json'), 'w') as f:
        json.dump(imm_token_counts_by_congress_by_party, f, indent=2)


if __name__ == '__main__':
    main()
