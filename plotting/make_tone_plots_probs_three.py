import os
import json
from optparse import OptionParser
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from time_periods.common import periods, congress_to_year
from plotting.make_tone_plots import plot_bg_fill, combine_name_state_and_party, plot_percent_diff_line_with_bands, add_labels_to_plot

# Combine CFM plots from three different approaches

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--pred-dir', type=str, default='data/speeches/Congress/predictions_cfm/',
                      help='Directory with collected predictions using a combined cfm: default=%default')
    parser.add_option('--pred-dir-rep', type=str, default='data/speeches/Congress/predictions_cfm_R/',
                      help='Directory with collected predictions using a modern R cfm): default=%default')
    parser.add_option('--pred-dir-dem', type=str, default='data/speeches/Congress/predictions_cfm_D/',
                      help='Directory with collected predictions using a modern D cfm: default=%default')
    parser.add_option('--pres-file', type=str, default='data/speeches/Congress/predictions_cfm/pres_imm_segments_with_tone.jsonlist',
                      help='.jsonlist file with presidential segments and tone: default=%default')
    parser.add_option('--pres-counts-file', type=str, default='data/speeches/Presidential/paragraph_counts.json',
                      help='.json file with presidential paragraph counts (from presidential.export_presidential_segments.py): default=%default')
    parser.add_option('--procedural-file', type=str, default='data/speeches/Congress/procedural_speech_ids.txt',
                      help='File with list of procdeural speeches: default=%default')
    parser.add_option('--counts-dir', type=str, default='data/speeches/Congress/basic_counts/',
                      help='Directory with stored speech counts (from analysis.count_speeches_and_tokens.py: default=%default')
    parser.add_option('--country-dir', type=str, default='data/speeches/Congress/country_mentions/',
                      help='Directory with counts by nationality: default=%default')
    parser.add_option('--outdir', type=str, default='plots_probs/',
                      help='Output dir: default=%default')
    parser.add_option('--first-congress', type=int, default=46,
                      help='First congress: default=%default')
    parser.add_option('--last-congress', type=int, default=116,
                      help='Last congress: default=%default')
    parser.add_option('--imm-thresh', type=float, default=0.9,
                      help='Require p(immigration) above this threshold: default=%default')
    parser.add_option('--tone-thresh', type=float, default=0.9,
                      help='Require p(tone) above this threshold: default=%default')
    #parser.add_option('--samples', type=int, default=30,
    #                  help='Number of samples: default=%default')
    #parser.add_option('--only-keyword', action="store_true", default=False,
    #                  help='Only use keyword speeches: default=%default')

    (options, args) = parser.parse_args()

    pred_dir = options.pred_dir
    pred_dir_rep = options.pred_dir_rep
    pred_dir_dem = options.pred_dir_dem
    tone_file = os.path.join(pred_dir, 'imm_speech_ids_with_tone.tsv')
    labeled_dict_file = os.path.join(pred_dir, 'tone_by_speech_id.json')
    tone_file_rep = os.path.join(pred_dir_rep, 'imm_speech_ids_with_tone.tsv')
    labeled_dict_file_rep = os.path.join(pred_dir_rep, 'tone_by_speech_id.json')    
    tone_file_dem = os.path.join(pred_dir_dem, 'imm_speech_ids_with_tone.tsv')
    labeled_dict_file_dem = os.path.join(pred_dir_dem, 'tone_by_speech_id.json')
    pres_file = options.pres_file
    pres_counts_file = options.pres_counts_file
    procedural_file = options.procedural_file
    counts_dir = options.counts_dir
    country_dir = options.country_dir
    outdir = options.outdir
    first_congress = options.first_congress
    last_congress = options.last_congress
    imm_thresh = options.imm_thresh
    tone_thresh = options.tone_thresh
    #n_samples = options.samples
    #only_keyword = options.only_keyword

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    congress_range = list(range(first_congress, last_congress+1))
    years = [congress_to_year(c) for c in congress_range]
    print(years[0], years[-1])

    print("Loading country mentions")
    with open(os.path.join(country_dir, 'imm_country_speech_ids_by_nationality_or_country_mentions.json')) as f:
        speech_ids_by_country = json.load(f)
    with open(os.path.join(country_dir, 'imm_country_counts_nationality_mentions.json')) as f:
        nationality_mentions = json.load(f)

    # exclude India and England because their numbers are not reliable
    countries = sorted([g for g in nationality_mentions if g != 'India' and g != 'England' and g != 'Canada' and g != 'France'])
    counts = [nationality_mentions[g] for g in countries]
    order = np.argsort(counts)[::-1]
    target_countries = [countries[i] for i in order[:20]]
    target_countries_set = set(target_countries)
    print("Target nationalities:")
    print(target_countries)

    print("Loading presidential data")
    with open(pres_file) as f:
        lines = f.readlines()
    pres_segments = [json.loads(line) for line in lines]
    print(len(pres_segments))

    with open(pres_counts_file) as f:
        pres_counts = json.load(f)

    print("Loading speech data")
    df = pd.read_csv(tone_file, header=0, index_col=0, sep='\t')
    print(df.shape)

    df_rep = pd.read_csv(tone_file_rep, header=0, index_col=0, sep='\t')
    print(df_rep.shape)

    df_dem = pd.read_csv(tone_file_dem, header=0, index_col=0, sep='\t')
    print(df_dem.shape)

    with open(labeled_dict_file) as f:
        labels_dict = json.load(f)

    with open(labeled_dict_file_rep) as f:
        labels_dict_rep = json.load(f)

    with open(labeled_dict_file_dem) as f:
        labels_dict_dem = json.load(f)

    print("Filtering speeches")
    # start with the 1881 congress
    df = df[df['congress'] >= first_congress]
    df_rep = df_rep[df_rep['congress'] >= first_congress]
    df_dem = df_dem[df_dem['congress'] >= first_congress]

    print(df.shape)

    # Load speeches to exclude
    print("Loading procedural speech ids")
    with open(procedural_file) as f:
        to_exclude = f.readlines()
    to_exclude = set([s.strip() for s in to_exclude])
    print(len(to_exclude))

    print("Loading speech counts")
    with open(os.path.join(counts_dir, 'tokens_by_congress.json')) as f:
        tokens_by_congress = json.load(f)
    tokens_by_congress = {int(congress): int(count) for congress, count in tokens_by_congress.items()}

    with open(os.path.join(counts_dir, 'tokens_by_congress_by_party.json')) as f:
        tokens_by_congress_by_party = json.load(f)
    for party in ['D', 'R']:
        tokens_by_congress_by_party[party] = {int(congress): int(count) for congress, count in tokens_by_congress_by_party[party].items()}

    print("Loading speech counts")
    with open(os.path.join(counts_dir, 'imm_tokens_by_congress.json')) as f:
        imm_tokens_by_congress = json.load(f)
    imm_tokens_by_congress = {int(congress): int(count) for congress, count in imm_tokens_by_congress.items()}

    with open(os.path.join(counts_dir, 'imm_tokens_by_congress_by_party.json')) as f:
        imm_tokens_by_congress_by_party = json.load(f)
    for party in ['D', 'R']:
        imm_tokens_by_congress_by_party[party] = {int(congress): int(count) for congress, count in imm_tokens_by_congress_by_party[party].items()}

    imm_speech_id_set = set(df['speech_id'].values)
    imm_speech_id_list = list(df['speech_id'].values)

    congresses = list(df['congress'].values)
    parties = list(df['party'].values)
    speakers = list(df['speaker'].values)
    states = list(df['state'].values)
    chambers = list(df['chamber'].values)
    imm_speech_id_list = list(df['speech_id'].values)
    imm_probs = list(df['imm_prob'].values)

    tone_scores = np.array(df[['anti_prob_sum', 'neutral_prob_sum', 'pro_prob_sum']].values)
    tone_probs = tone_scores / tone_scores.sum(1).reshape((len(tone_scores), 1))
    tones = tone_scores.argmax(1)

    print("Identifying top speakers")
    party_by_speaker = {}
    speaker_counter = Counter()
    for i, speaker in enumerate(speakers):
        name = combine_name_state_and_party(speaker, parties[i], states[i])
        party_by_speaker[name] = parties[i]
        speaker_counter[name] += 1
    target_speakers = []
    for speaker, count in speaker_counter.most_common(n=100):
        if not speaker.startswith('The'):
            target_speakers.append(speaker)
    for speaker in target_speakers[:20]:
        print(speaker, speaker_counter[speaker])
    top_20_speakers = target_speakers[:20]

    imm_speeches_per_congress = Counter()
    pro_imm_speeches_per_congress = Counter()
    anti_imm_speeches_per_congress = Counter()
    neutral_imm_speeches_per_congress = Counter()

    imm_speeches_per_congress_by_party = defaultdict(Counter)
    pro_imm_speeches_per_congress_by_party = defaultdict(Counter)
    neutral_imm_speeches_per_congress_by_party = defaultdict(Counter)
    anti_imm_speeches_per_congress_by_party = defaultdict(Counter)

    # Collect presidential data
    tones_by_president = defaultdict(Counter)
    pro_tones_by_president = defaultdict(Counter)
    anti_tones_by_president = defaultdict(Counter)
    imm_docs_per_year_by_president = {}

    print("Collecting presidential data")
    for segment in tqdm(pres_segments):
        segment_id = segment['id']
        speech_id = '_'.join(segment_id.split('_')[:-1])
        president = segment['speaker']
        year = int(segment['year'])
        pres_tone_probs = np.array([segment['anti_prob'], segment['neutral_prob'], segment['pro_prob']])
        #imm_segments_by_president[president] += 1
        tone = np.argmax(pres_tone_probs)
        #tones_by_president[president] += 1
        tones_by_president[president][year] += 1
        if tone == 0:
            #anti_tones_by_president[president] += 1
            anti_tones_by_president[president][year] += 1
        elif tone == 2:
            #pro_tones_by_president[president] += 1
            pro_tones_by_president[president][year] += 1
        if president not in imm_docs_per_year_by_president:
            imm_docs_per_year_by_president[president] = defaultdict(set)
        imm_docs_per_year_by_president[president][year].add(speech_id)

    party_to_color = {'D': 'blue', 'R': 'red'}
    
    # make overall tone plot / main tone plot

    print("Making basic tone plot")
    #fig, axes = plt.subplots(nrows=3, figsize=(12, 9.5))
    nrows = 2
    fig, axes = plt.subplots(nrows=nrows, figsize=(12, 6.3))
    plt.subplots_adjust(hspace=0.3)

    lower = -105
    upper = 105
    for ax in range(nrows):
        plot_bg_fill(axes[ax], periods, lower, upper, party_to_color)


    # aggregate the tones per speech along various dimensions
    n_excluded = 0
    for i, congress in tqdm(enumerate(congresses), total=len(congresses)):
        speech_id = str(imm_speech_id_list[i])
        chamber = chambers[i]
        if speech_id in to_exclude:
            n_excluded += 1
        else:
            imm_speeches_per_congress[congress] += 1
            imm_speeches_per_congress_by_party[parties[i]][congress] += 1

            if tones[i] == 0:
                anti_imm_speeches_per_congress[congress] += 1
                anti_imm_speeches_per_congress_by_party[parties[i]][congress] += 1
                #anti_imm_speeches_per_congress_by_region[state_to_region(states[i])][congress] += 1

            elif tones[i] == 2:
                pro_imm_speeches_per_congress[congress] += 1
                pro_imm_speeches_per_congress_by_party[parties[i]][congress] += 1
                #pro_imm_speeches_per_congress_by_region[state_to_region(states[i])][congress] += 1

            else:
                neutral_imm_speeches_per_congress[congress] += 1
                neutral_imm_speeches_per_congress_by_party[parties[i]][congress] += 1
                #neutral_imm_speeches_per_congress_by_region[state_to_region(states[i])][congress] += 1

    print(n_excluded, 'excluded')
    print(sum(imm_speeches_per_congress.values()))


    ax = 0  # by party
    plot_percent_diff_line_with_bands(axes[ax], pro_imm_speeches_per_congress, anti_imm_speeches_per_congress, imm_speeches_per_congress, congress_range, years, color='k', label='All speeches', fill_alpha=0.1, line_alpha=0.5, linestyle='-.')
    party = 'D'
    plot_percent_diff_line_with_bands(axes[ax], pro_imm_speeches_per_congress_by_party[party], anti_imm_speeches_per_congress_by_party[party], imm_speeches_per_congress_by_party[party], congress_range, years, party_to_color[party], label='Democrat')
    party = 'R'
    plot_percent_diff_line_with_bands(axes[ax], pro_imm_speeches_per_congress_by_party[party], anti_imm_speeches_per_congress_by_party[party], imm_speeches_per_congress_by_party[party], congress_range, years, party_to_color[party], label='Republican')

    add_labels_to_plot(axes[ax], years, lower, upper, ylabel='% Pro - % Anti speeches', title='Net tone of immigration speeches in Congress by party')

    # by president
    ax = 1
    plot_percent_diff_line_with_bands(axes[ax], pro_imm_speeches_per_congress, anti_imm_speeches_per_congress, imm_speeches_per_congress, congress_range, years, 'k', label=None, fill_alpha=0.1, line_alpha=0.5, linestyle='-.')
    # presidents to plot per congress rather than overall
    recent_presidents = {'Donald J. Trump', 'Barack Obama', 'George W. Bush', 'William J. Clinton', 'George Bush',
                         'Ronald Reagan', 'Jimmy Carter', 'Gerald R. Ford', 'Richard Nixon', 'Lyndon B. Johnson',
                         'John F. Kennedy', 'Dwight D. Eisenhower'}

    for start, end, party, person, nickname in periods:
        president_years = range(start, end, 1)
        print(person, start, end, president_years)
        if person in recent_presidents:
            # plot tone by year
            pres_pro_by_congress = {y: pro_tones_by_president[person][y] for y in president_years}
            pres_anti_by_congress = {y: anti_tones_by_president[person][y] for y in president_years}
            n_tones_by_congress = {y: tones_by_president[person][y] for y in president_years}
            #plot_percent_line_with_bands(axes[ax], tone_sum_by_year, n_tones_by_year, president_years, [y-0.5 for y in president_years], party_to_color[party], label=None)
        else:
            # just plot a single mean over the entire time period
            n_pro = 0
            n_anti = 0
            n_tones = 0
            for y in president_years:
                n_pro += pro_tones_by_president[person][y]
                n_anti += anti_tones_by_president[person][y]
                n_tones += tones_by_president[person][y]
            pres_pro_by_congress = {y: n_pro for y in president_years}
            pres_anti_by_congress = {y: n_anti for y in president_years}
            n_tones_by_congress = {y: n_tones for y in president_years}
        #print(person, [n_tones_by_congress[y]for y in president_congresses])
        plot_percent_diff_line_with_bands(axes[ax], pres_pro_by_congress, pres_anti_by_congress, n_tones_by_congress, president_years, president_years, party_to_color[party], label=None)
        #scatter_percent_diff(axes[ax], pres_pro_by_congress, pres_anti_by_congress, n_tones_by_congress, president_years, president_years, party_to_color[party], label=None)
        mean_tone = (sum(pro_tones_by_president[person].values()) - sum(anti_tones_by_president[person].values())) / sum(tones_by_president[person].values())

        if start > 1880:
            if mean_tone > 0:
                axes[ax].text(start + (end-start)/2-1.5, lower+10, nickname, rotation=90, va='bottom', alpha=0.75)
            else:
                axes[ax].text(start + (end-start)/2-1.5, upper-10, nickname, rotation=90, va='top', alpha=0.75)
    add_labels_to_plot(axes[ax], years, lower, upper, ylabel='% Pro - % Anti speeches', title='Net tone of immigration speeches by President', legend_loc=None)

    plt.savefig(os.path.join(outdir, 'main_tone_plot.png'), bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'main_tone_plot.pdf'), bbox_inches='tight')



    # Make plot with average probabilities, rather than thresholded labels

    # redo with sampling tones from each segment
    print("Making sampling tone plot")
    nrows = 2
    fig, axes = plt.subplots(nrows=nrows, figsize=(12, 6.3))
    plt.subplots_adjust(hspace=0.3)

    lower = -105
    upper = 105
    for ax in range(nrows):
        plot_bg_fill(axes[ax], periods, lower, upper, party_to_color)

    # aggregate the tones per speech along various dimensions
    imm_speeches_per_congress = Counter()
    pro_imm_prob_sums_per_congress = Counter()
    anti_imm_prob_sums_per_congress = Counter()
    neutral_imm_prob_sums_per_congress = Counter()

    n_excluded = 0
    for i, congress in tqdm(enumerate(congresses), total=len(congresses)):
        speech_id = str(imm_speech_id_list[i])

        if speech_id in to_exclude:
            n_excluded += 1
        else:
            imm_speeches_per_congress[congress] += 1
            imm_speeches_per_congress_by_party[parties[i]][congress] += 1

            data = labels_dict[speech_id]
            if 'keyword_probs' in data:
                probs = data['keyword_probs']
            else:
                probs = data['non_keyword_probs']

            tone_sums = np.zeros(3)
            for prob_vector in probs:                    
                tone_sums += prob_vector
            mean_tone_probs = tone_sums / tone_sums.sum()

            anti_imm_prob_sums_per_congress[congress] += mean_tone_probs[0]

            neutral_imm_prob_sums_per_congress[congress] += mean_tone_probs[1]

            pro_imm_prob_sums_per_congress[congress] += mean_tone_probs[2]

    ax = 0  # by party
    plot_percent_diff_line_with_bands(axes[ax], pro_imm_prob_sums_per_congress, anti_imm_prob_sums_per_congress, imm_speeches_per_congress, congress_range, years, 'k', label='All speeches', fill_alpha=0.1, line_alpha=0.5, linestyle='-.', bands=True)


    # aggregate the tones per speech along various dimensions
    imm_speeches_per_congress_by_party = defaultdict(Counter)
    pro_imm_prob_sums_per_congress_by_party = defaultdict(Counter)
    neutral_imm_prob_sums_per_congress_by_party = defaultdict(Counter)
    anti_imm_prob_sums_per_congress_by_party = defaultdict(Counter)        

    n_excluded = 0
    for i, congress in tqdm(enumerate(congresses), total=len(congresses)):
        speech_id = str(imm_speech_id_list[i])

        if speech_id in to_exclude:
            n_excluded += 1
        elif parties[i] == 'R':
            imm_speeches_per_congress_by_party[parties[i]][congress] += 1

            data = labels_dict_rep[speech_id]
            if 'keyword_probs' in data:
                probs = data['keyword_probs']
            else:
                probs = data['non_keyword_probs']

            tone_sums = np.zeros(3)
            for prob_vector in probs:                    
                tone_sums += prob_vector
            mean_tone_probs = tone_sums / tone_sums.sum()

            anti_imm_prob_sums_per_congress_by_party[parties[i]][congress] += mean_tone_probs[0]

            neutral_imm_prob_sums_per_congress_by_party[parties[i]][congress] += mean_tone_probs[1]

            pro_imm_prob_sums_per_congress_by_party[parties[i]][congress] += mean_tone_probs[2]


    party = 'R'
    plot_percent_diff_line_with_bands(axes[ax], pro_imm_prob_sums_per_congress_by_party[party], anti_imm_prob_sums_per_congress_by_party[party], imm_speeches_per_congress_by_party[party], congress_range, years, party_to_color[party], label=party, line_alpha=1.0, bands=True)



    # aggregate the tones per speech along various dimensions
    imm_speeches_per_congress_by_party = defaultdict(Counter)
    pro_imm_prob_sums_per_congress_by_party = defaultdict(Counter)
    neutral_imm_prob_sums_per_congress_by_party = defaultdict(Counter)
    anti_imm_prob_sums_per_congress_by_party = defaultdict(Counter)        

    n_excluded = 0
    for i, congress in tqdm(enumerate(congresses), total=len(congresses)):
        speech_id = str(imm_speech_id_list[i])

        if speech_id in to_exclude:
            n_excluded += 1
        elif parties[i] == 'D':
            imm_speeches_per_congress_by_party[parties[i]][congress] += 1

            data = labels_dict_dem[speech_id]
            if 'keyword_probs' in data:
                probs = data['keyword_probs']
            else:
                probs = data['non_keyword_probs']

            tone_sums = np.zeros(3)
            for prob_vector in probs:                    
                tone_sums += prob_vector
            mean_tone_probs = tone_sums / tone_sums.sum()

            anti_imm_prob_sums_per_congress_by_party[parties[i]][congress] += mean_tone_probs[0]

            neutral_imm_prob_sums_per_congress_by_party[parties[i]][congress] += mean_tone_probs[1]

            pro_imm_prob_sums_per_congress_by_party[parties[i]][congress] += mean_tone_probs[2]


    party = 'D'
    plot_percent_diff_line_with_bands(axes[ax], pro_imm_prob_sums_per_congress_by_party[party], anti_imm_prob_sums_per_congress_by_party[party], imm_speeches_per_congress_by_party[party], congress_range, years, party_to_color[party], label=party, line_alpha=1.0, bands=True)


    # Collect presidential data
    tones_by_president = defaultdict(Counter)
    pro_tones_by_president = defaultdict(Counter)
    anti_tones_by_president = defaultdict(Counter)
    imm_docs_per_year_by_president = {}

    for segment in tqdm(pres_segments):
        segment_id = segment['id']
        speech_id = '_'.join(segment_id.split('_')[:-1])
        president = segment['speaker']
        imm_prob = segment['imm_prob']
        year = int(segment['year'])
        pres_tone_probs = np.array([segment['anti_prob'], segment['neutral_prob'], segment['pro_prob']])
        pres_tone_probs = pres_tone_probs / pres_tone_probs.sum()
        tones_by_president[president][year] += 1
        anti_tones_by_president[president][year] += pres_tone_probs[0]
        pro_tones_by_president[president][year] += pres_tone_probs[2]
        if president not in imm_docs_per_year_by_president:
            imm_docs_per_year_by_president[president] = defaultdict(set)
        imm_docs_per_year_by_president[president][year].add(speech_id)


    # by president
    ax = 1
    #plot_percent_diff_line_with_bands(axes[ax], pro_imm_speeches_per_congress, anti_imm_speeches_per_congress, imm_speeches_per_congress, congress_range, years, 'k', label=None, fill_alpha=0.1, line_alpha=0.5, linestyle='-.')
    # presidents to plot per congress rather than overall
    recent_presidents = {'Donald J. Trump', 'Barack Obama', 'George W. Bush', 'William J. Clinton', 'George Bush',
                        'Ronald Reagan', 'Jimmy Carter', 'Gerald R. Ford', 'Richard Nixon', 'Lyndon B. Johnson',
                        'John F. Kennedy', 'Dwight D. Eisenhower'}

    for start, end, party, person, nickname in periods:
        president_years = range(start, end, 1)
        #print(person, start, end, president_years)
        if person in recent_presidents:
            # plot tone by year
            pres_pro_by_congress = {y: pro_tones_by_president[person][y] for y in president_years}
            pres_anti_by_congress = {y: anti_tones_by_president[person][y] for y in president_years}
            n_tones_by_congress = {y: tones_by_president[person][y] for y in president_years}
            #plot_percent_line_with_bands(axes[ax], tone_sum_by_year, n_tones_by_year, president_years, [y-0.5 for y in president_years], party_to_color[party], label=None)
        else:
            # just plot a single mean over the entire time period
            n_pro = 0
            n_anti = 0
            n_tones = 0
            for y in president_years:
                n_pro += pro_tones_by_president[person][y]
                n_anti += anti_tones_by_president[person][y]
                n_tones += tones_by_president[person][y]
            pres_pro_by_congress = {y: n_pro for y in president_years}
            pres_anti_by_congress = {y: n_anti for y in president_years}
            n_tones_by_congress = {y: n_tones for y in president_years}
        #print(person, [n_tones_by_congress[y]for y in president_congresses])
        plot_percent_diff_line_with_bands(axes[ax], pres_pro_by_congress, pres_anti_by_congress, n_tones_by_congress, president_years, president_years, party_to_color[party], label=None)
        #scatter_percent_diff(axes[ax], pres_pro_by_congress, pres_anti_by_congress, n_tones_by_congress, president_years, president_years, party_to_color[party], label=None)
        mean_tone = (sum(pro_tones_by_president[person].values()) - sum(anti_tones_by_president[person].values())) / sum(tones_by_president[person].values())
        
        if start > 1880:
            if mean_tone > 0:
                axes[ax].text(start + (end-start)/2-1.5, lower+10, nickname, rotation=90, va='bottom', alpha=0.75)
            else:
                axes[ax].text(start + (end-start)/2-1.5, upper-10, nickname, rotation=90, va='top', alpha=0.75)

    add_labels_to_plot(axes[0], years, lower, upper, ylabel='% Pro - % Anti speeches', title='Net tone of immigration speeches in Congress by party')
    add_labels_to_plot(axes[1], years, lower, upper, ylabel='% Pro - % Anti speeches', title='Net tone of immigration speeches by President', legend_loc=None)

    plt.savefig(os.path.join(outdir, 'main_tone_plot_probs_three.png'), bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'main_tone_plot_probs_three.pdf'), bbox_inches='tight')



if __name__ == '__main__':
    main()
