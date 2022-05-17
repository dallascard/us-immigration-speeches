import os
import json
from optparse import OptionParser
from collections import defaultdict, Counter

import seaborn
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from time_periods.common import periods, year_to_congress, congress_to_year, congress_to_decade


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--tone-file', type=str, default='data/speeches/Congress/imm_speech_ids_with_tone.tsv',
                      help='.tsv file with immigration segments and tones: default=%default')
    parser.add_option('--pres-file', type=str, default='data/speeches/Presidential/pres_imm_segments_with_tone.jsonlist',
                      help='.jsonlist file with presidential segments and tone: default=%default')
    parser.add_option('--pres-counts-file', type=str, default='data/speeches/Presidential/paragraph_counts.json',
                      help='.json file with presidential paragraph counts (from presidential.export_presidential_segments.py): default=%default')
    parser.add_option('--procedural-file', type=str, default='data/speeches/Congress/procedural_speech_ids.txt',
                      help='File with list of procdeural speeches: default=%default')
    parser.add_option('--counts-dir', type=str, default='data/speeches/Congress/basic_counts/',
                      help='Directory with stored speech counts (from analysis.count_speeches_and_tokens.py: default=%default')
    parser.add_option('--country-dir', type=str, default='data/speeches/Congress/country_mentions/',
                      help='Directory with counts by nationality: default=%default')
    parser.add_option('--outdir', type=str, default='plots/',
                      help='Output dir: default=%default')
    parser.add_option('--first-congress', type=int, default=46,
                      help='First congress: default=%default')
    parser.add_option('--last-congress', type=int, default=116,
                      help='Last congress: default=%default')
    parser.add_option('--only-keyword', action="store_true", default=False,
                      help='Only use keyword speeches: default=%default')

    (options, args) = parser.parse_args()

    tone_file = options.tone_file
    pres_file = options.pres_file
    pres_counts_file = options.pres_counts_file
    procedural_file = options.procedural_file
    counts_dir = options.counts_dir
    country_dir = options.country_dir
    outdir = options.outdir
    first_congress = options.first_congress
    last_congress = options.last_congress
    only_keyword = options.only_keyword

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

    print("Filtering speeches")
    # start with the 1881 congress
    df = df[df['congress'] >= first_congress]

    if only_keyword:
        df = df[df['keyword'] == 1]

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
    imm_speech_id_list = list(df['speech_id'].values)

    tone_scores = np.array(df[['anti_prob_sum', 'neutral_prob_sum', 'pro_prob_sum']].values)
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
    top_20_speaker_set = set(top_20_speakers)

    imm_speeches_per_congress = Counter(congresses)
    imm_speeches_per_decade = Counter(congresses)
    pro_imm_speeches_per_congress = Counter()
    anti_imm_speeches_per_congress = Counter()
    neutral_imm_speeches_per_congress = Counter()

    imm_speeches_per_congress_by_party = defaultdict(Counter)
    pro_imm_speeches_per_congress_by_party = defaultdict(Counter)
    neutral_imm_speeches_per_congress_by_party = defaultdict(Counter)
    anti_imm_speeches_per_congress_by_party = defaultdict(Counter)

    imm_speeches_per_congress_by_country = defaultdict(Counter)
    pro_imm_speeches_per_congress_by_country = defaultdict(Counter)
    neutral_imm_speeches_per_congress_by_country = defaultdict(Counter)
    anti_imm_speeches_per_congress_by_country = defaultdict(Counter)

    imm_speeches_per_decade_by_speaker = defaultdict(Counter)
    pro_imm_speeches_per_decade_by_speaker = defaultdict(Counter)
    neutral_imm_speeches_per_decade_by_speaker = defaultdict(Counter)
    anti_imm_speeches_per_decade_by_speaker = defaultdict(Counter)

    imm_speeches_per_congress_by_speaker = defaultdict(Counter)
    pro_imm_speeches_per_congress_by_speaker = defaultdict(Counter)
    neutral_imm_speeches_per_congress_by_speaker = defaultdict(Counter)
    anti_imm_speeches_per_congress_by_speaker = defaultdict(Counter)

    # aggregate the tones per speech along various dimensions
    n_excluded = 0
    for i, congress in tqdm(enumerate(congresses), total=len(congresses)):
        speech_id = str(imm_speech_id_list[i])
        if speech_id in to_exclude:
            n_excluded += 1
        else:
            speaker = combine_name_state_and_party(speakers[i], parties[i], states[i])

            # get the decade as the year ending (i.e. 2011-2020 -> 2020)
            decade = congress_to_decade(congress)

            imm_speeches_per_decade[decade] += 1
            imm_speeches_per_congress_by_party[parties[i]][congress] += 1

            if speaker in top_20_speaker_set:
                imm_speeches_per_decade_by_speaker[speaker][decade] += 1
                imm_speeches_per_congress_by_speaker[speaker][congress] += 1

            countries_mentioned = []

            # determine the countries mentioned in this speech
            for country in target_countries_set:
                if speech_id in speech_ids_by_country[country]:
                    countries_mentioned.append(country)
                    imm_speeches_per_congress_by_country[country][congress] += 1

            if tones[i] == 0:
                anti_imm_speeches_per_congress[congress] += 1
                anti_imm_speeches_per_congress_by_party[parties[i]][congress] += 1
                for country in countries_mentioned:
                    anti_imm_speeches_per_congress_by_country[country][congress] += 1
                if speaker in top_20_speakers:
                    anti_imm_speeches_per_decade_by_speaker[speaker][decade] += 1
                    anti_imm_speeches_per_congress_by_speaker[speaker][congress] += 1
            elif tones[i] == 2:
                pro_imm_speeches_per_congress[congress] += 1
                pro_imm_speeches_per_congress_by_party[parties[i]][congress] += 1
                for country in countries_mentioned:
                    pro_imm_speeches_per_congress_by_country[country][congress] += 1
                if speaker in top_20_speakers:
                    pro_imm_speeches_per_decade_by_speaker[speaker][decade] += 1
                    pro_imm_speeches_per_congress_by_speaker[speaker][congress] += 1
            else:
                neutral_imm_speeches_per_congress[congress] += 1
                neutral_imm_speeches_per_congress_by_party[parties[i]][congress] += 1
                for country in countries_mentioned:
                    neutral_imm_speeches_per_congress_by_country[country][congress] += 1
                if speaker in top_20_speakers:
                    neutral_imm_speeches_per_decade_by_speaker[speaker][decade] += 1
                    neutral_imm_speeches_per_congress_by_speaker[speaker][congress] += 1

    print(n_excluded, 'excluded')
    print(sum(imm_speeches_per_congress.values()))

    print("Combining R and D")
    # add up the counts for D and R speeches (ignoring others)
    imm_speeches_per_congress_either = {y: imm_speeches_per_congress_by_party['D'][y] + imm_speeches_per_congress_by_party['R'][y] for y in congress_range}
    pro_imm_speeches_per_congress_either = {y: pro_imm_speeches_per_congress_by_party['D'][y] + pro_imm_speeches_per_congress_by_party['R'][y] for y in congress_range}
    neutral_imm_speeches_per_congress_either = {y: neutral_imm_speeches_per_congress_by_party['D'][y] + neutral_imm_speeches_per_congress_by_party['R'][y] for y in congress_range}
    anti_imm_speeches_per_congress_either = {y: anti_imm_speeches_per_congress_by_party['D'][y] + anti_imm_speeches_per_congress_by_party['R'][y] for y in congress_range}

    net_pro_speeches_per_congress = {c: pro_imm_speeches_per_congress[c] - anti_imm_speeches_per_congress[c] for c in congress_range}

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
        tone_probs = np.array([segment['anti_prob'], segment['neutral_prob'], segment['pro_prob']])
        tone = np.argmax(tone_probs)
        tones_by_president[president][year] += 1
        if tone == 0:
            anti_tones_by_president[president][year] += 1
        elif tone == 2:
            pro_tones_by_president[president][year] += 1
        if president not in imm_docs_per_year_by_president:
            imm_docs_per_year_by_president[president] = defaultdict(set)
        imm_docs_per_year_by_president[president][year].add(speech_id)

    party_to_color = {'D': 'blue', 'R': 'red'}

    # Plot % Pro, Neutral, Anti by party (and overall)
    print("Making pro/neutral/anti plot")
    seaborn.set_palette(seaborn.color_palette("tab10"))
    fig, axes = plt.subplots(nrows=3, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.4)

    lower = 0
    upper = 100

    ax = 0
    plot_bg_fill(axes[ax], periods, lower, upper, party_to_color)
    plot_percent_line_with_bands(axes[ax], pro_imm_speeches_per_congress, imm_speeches_per_congress, congress_range, years, 'C2', '% Pro')
    plot_percent_line_with_bands(axes[ax], neutral_imm_speeches_per_congress, imm_speeches_per_congress, congress_range, years, 'grey', '% Neutral')
    plot_percent_line_with_bands(axes[ax], anti_imm_speeches_per_congress, imm_speeches_per_congress, congress_range, years, 'C1', '% Anti')
    add_labels_to_plot(axes[ax], years, lower, upper, ylabel='% of speeches', title='Percent of immigration speeches by tone (all)')

    ax = 1
    party = 'D'
    plot_bg_fill(axes[ax], periods, lower, upper, party_to_color)
    plot_percent_line_with_bands(axes[ax], pro_imm_speeches_per_congress_by_party[party], imm_speeches_per_congress_by_party[party], congress_range, years, 'C2', '% Pro (D)')
    plot_percent_line_with_bands(axes[ax], neutral_imm_speeches_per_congress_by_party[party], imm_speeches_per_congress_by_party[party], congress_range, years, 'grey', '% Neutral (D)')
    plot_percent_line_with_bands(axes[ax], anti_imm_speeches_per_congress_by_party[party], imm_speeches_per_congress_by_party[party], congress_range, years, 'C1', '% Anti (D)')
    add_labels_to_plot(axes[ax], years, lower, upper, ylabel='% of speeches', title='Percent of immigration speeches by tone (Democrats)')

    ax = 2
    party = 'R'
    plot_bg_fill(axes[ax], periods, lower, upper, party_to_color)
    plot_percent_line_with_bands(axes[ax], pro_imm_speeches_per_congress_by_party[party], imm_speeches_per_congress_by_party[party], congress_range, years, 'C2', '% Pro (R)')
    plot_percent_line_with_bands(axes[ax], neutral_imm_speeches_per_congress_by_party[party], imm_speeches_per_congress_by_party[party], congress_range, years, 'grey', '% Neutral (R)')
    plot_percent_line_with_bands(axes[ax], anti_imm_speeches_per_congress_by_party[party], imm_speeches_per_congress_by_party[party], congress_range, years, 'C1', '% Anti (R)')
    add_labels_to_plot(axes[ax], years, lower, upper, ylabel='% of speeches', title='Percent of immigration speeches by tone (Republicans)')

    plt.savefig(os.path.join(outdir, 'tone_percentages_by_party.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'tone_percentages_by_party.png'), bbox_inches='tight')


    # make plots by speaker
    lower = -100
    upper = 100

    seaborn.set_palette(seaborn.color_palette("tab20"))
    fig, axes = plt.subplots(ncols=2, nrows=10, figsize=(12, 12), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.15)
    plt.subplots_adjust(wspace=0.02)

    speaker_start_congresses = {}
    for g_ii, speaker in enumerate(top_20_speakers):
        valid_congresses = [year for year, count in imm_speeches_per_congress_by_speaker[speaker].items() if count >= 10]
        speaker_start_congresses[speaker] = min(valid_congresses)

    speakers_to_plot = sorted(speaker_start_congresses)
    start_congresses = [speaker_start_congresses[s] for s in speakers_to_plot]
    order = np.argsort(start_congresses)

    for g_i in range(20):
        speaker = speakers_to_plot[order[g_i]]
        print(speaker, speaker_start_congresses[speaker])
        if ' of ' in speaker:
            parts = speaker.split()
            name = ' '.join(parts[:2] + parts[-2:])
        else:
            name = speaker
        row = g_i
        col = 0
        if g_i >= 10:
            row -= 10
            col = 1

        plot_percent_diff_line_with_bands(axes[row][col], pro_imm_speeches_per_congress, anti_imm_speeches_per_congress, imm_speeches_per_congress, congress_range, years, 'k', label=None, line_alpha=0.8)

        valid_congresses = [year for year, count in imm_speeches_per_congress_by_speaker[speaker].items() if count >= 10]

        # fix an error (or duplicate name?)
        if speaker == 'Mr. LODGE (R, MA)':
            valid_congresses = [v for v in valid_congresses if v < 76]
        elif speaker == 'Mr. KENNEDY (D, MA)':
            valid_congresses = [v for v in valid_congresses if 86 < v < 112]
        xvals = [congress_to_year(d) for d in valid_congresses]
        plot_percent_diff_line_with_bands(axes[row][col], pro_imm_speeches_per_congress_by_speaker[speaker], anti_imm_speeches_per_congress_by_speaker[speaker], imm_speeches_per_congress_by_speaker[speaker], valid_congresses, xvals, party_to_color[party_by_speaker[speaker]], label=name)
        scatter_percent_diff(axes[row][col], pro_imm_speeches_per_congress_by_speaker[speaker], anti_imm_speeches_per_congress_by_speaker[speaker], imm_speeches_per_congress_by_speaker[speaker], valid_congresses, xvals, party_to_color[party_by_speaker[speaker]], label=None)
        add_labels_to_plot(axes[row][col], years, lower, upper)

    plt.savefig(os.path.join(outdir, 'tone_by_speaker_split.png'), bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'tone_by_speaker_split.pdf'), bbox_inches='tight')

    # make overall tone plot
    nrows = 2
    fig, axes = plt.subplots(nrows=nrows, figsize=(12, 6.3))
    plt.subplots_adjust(hspace=0.3)

    lower = -105
    upper = 105
    for ax in range(nrows):
        plot_bg_fill(axes[ax], periods, lower, upper, party_to_color)


    ax = 0  # by party
    plot_percent_diff_line_with_bands(axes[ax], pro_imm_speeches_per_congress, anti_imm_speeches_per_congress, imm_speeches_per_congress, congress_range, years, 'k', label='All speeches', fill_alpha=0.1, line_alpha=0.5, linestyle='-.')
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
        plot_percent_diff_line_with_bands(axes[ax], pres_pro_by_congress, pres_anti_by_congress, n_tones_by_congress, president_years, president_years, party_to_color[party], label=None)
        mean_tone = (sum(pro_tones_by_president[person].values()) - sum(anti_tones_by_president[person].values())) / sum(tones_by_president[person].values())

        if start > 1880:
            if mean_tone > 0:
                axes[ax].text(start + (end-start)/2-1.5, lower+10, nickname, rotation=90, va='bottom', alpha=0.75)
            else:
                axes[ax].text(start + (end-start)/2-1.5, upper-10, nickname, rotation=90, va='top', alpha=0.75)
    add_labels_to_plot(axes[ax], years, lower, upper, ylabel='% Pro - % Anti speeches', title='Net tone of immigration speeches by President', legend_loc=None)

    plt.savefig(os.path.join(outdir, 'main_tone_plot.png'), bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'main_tone_plot.pdf'), bbox_inches='tight')

    # make overall frequency plot
    fig, axes = plt.subplots(nrows=3, figsize=(12, 9))
    plt.subplots_adjust(hspace=0.3)

    lower = -0.5
    upper = 8
    for ax in range(3):
        plot_bg_fill(axes[ax], periods, lower, upper, party_to_color)

    ax = 0  # overall tone
    plot_percent_line_with_bands(axes[ax], imm_tokens_by_congress, tokens_by_congress, congress_range, years, 'k', label='All speeches')
    add_labels_to_plot(axes[ax], years, lower, upper, ylabel='% segments', title='Percent of non-procedural Congressional speeches relevant to immigration')

    ax = 1  # by party
    party = 'D'
    plot_percent_line_with_bands(axes[ax], imm_tokens_by_congress_by_party[party], tokens_by_congress_by_party[party], congress_range, years, party_to_color[party], label='Democrat')
    party = 'R'
    plot_percent_line_with_bands(axes[ax], imm_tokens_by_congress_by_party[party], tokens_by_congress_by_party[party], congress_range, years, party_to_color[party], label='Republican')
    add_labels_to_plot(axes[ax], years, lower, upper, ylabel='% segments', title='Percent of non-procedural Congressional speeches relevant to immigration by party')

    # by president
    ax = 2
    recent_presidents = {'Donald J. Trump', 'Barack Obama', 'George W. Bush', 'William J. Clinton', 'George Bush',
                         'Ronald Reagan', 'Jimmy Carter', 'Gerald R. Ford', 'Richard Nixon', 'Lyndon B. Johnson',
                         'John F. Kennedy', 'Dwight D. Eisenhower'}

    for start, end, party, person, nickname in periods:
        plain_name = person
        if person.startswith('Grover Cleveland'):
            plain_name = 'Grover Cleveland'
        president_years = range(start, end, 1)
        print(person, start, end, president_years)
        if person in recent_presidents:
            # plot tone by year
            n_tones_by_congress = {y: tones_by_president[person][y] for y in president_years}
            n_paragraphs_by_congress = {y: pres_counts[plain_name][str(y)] for y in president_years}
        else:
            # just plot a single mean over the entire time period
            n_tones = 0
            n_paragraphs = 0
            for y in president_years:
                n_tones += tones_by_president[person][y]
                n_paragraphs += pres_counts[plain_name][str(y)]
            n_tones_by_congress = {y: n_tones for y in president_years}
            n_paragraphs_by_congress = {y: n_paragraphs for y in president_years}
            print(person, n_tones, n_paragraphs, n_tones/n_paragraphs)
        plot_percent_line_with_bands(axes[ax], n_tones_by_congress, n_paragraphs_by_congress, president_years, president_years, party_to_color[party], label=None)
        if start > 1880:
            axes[2].text(start + (end-start)/2-1.5, upper-0.5, nickname, rotation=90, va='top', alpha=0.75)

    add_labels_to_plot(axes[ax], years, lower, upper, ylabel='% segments', title='Percent of Presidential communications relevant to immigration', legend_loc=None)

    plt.savefig(os.path.join(outdir, 'main_freq_plot.png'), bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'main_freq_plot.pdf'), bbox_inches='tight')

    # make plot with multiple countries

    lower = -105
    upper = 105

    seaborn.set_palette(seaborn.color_palette("tab10"))
    fig, ax = plt.subplots(figsize=(12, 3))

    plot_percent_diff_line_with_bands(ax, pro_imm_speeches_per_congress, anti_imm_speeches_per_congress, imm_speeches_per_congress, congress_range, years, 'k', label='All speeches', fill_alpha=0.1, line_alpha=0.5, linestyle='-.')

    to_plot = [target_countries[2], target_countries[1], target_countries[0]]
    for g_i, country in enumerate(to_plot):
        valid_congresses = sorted([year for year, count in imm_speeches_per_congress_by_country[country].items() if count >= 20])
        blocks = []
        for i, c in enumerate(valid_congresses):
            if i == 0:
                if c == valid_congresses[i+1]-1:
                    blocks.append([c])
            elif i == len(valid_congresses)-1:
                if c == valid_congresses[i-1]+1:
                    blocks[-1].append(c)
            elif c == valid_congresses[i+1]-1 or c == valid_congresses[i-1]+1:
                if len(blocks) == 0:
                    blocks.append([c])
                else:
                    blocks[-1].append(c)
            elif len(blocks) > 0 and len(blocks[-1]) > 0:
                blocks.append([])

        print(country, len(blocks))
        for b_i, block in enumerate(blocks):
            valid_congresses = block
            xvals = [congress_to_year(c) for c in valid_congresses]
            if b_i == 0:
                label = country
            else:
                label = None
            plot_percent_diff_line_with_bands(ax, pro_imm_speeches_per_congress_by_country[country], anti_imm_speeches_per_congress_by_country[country], imm_speeches_per_congress_by_country[country], valid_congresses, xvals, 'C' + str(g_i), label=label)

    add_labels_to_plot(ax, years, lower, upper, ylabel='% Pro - % Anti speeches')

    plt.savefig(os.path.join(outdir, 'tone_by_country_four.png'), bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'tone_by_country_four.pdf'), bbox_inches='tight')


    #  Make plot with top 14 countries

    lower = -100
    upper = 100

    seaborn.set_palette(seaborn.color_palette("tab10"))
    nrows = 7
    fig, axes = plt.subplots(ncols=2, nrows=nrows, figsize=(12, 9), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.15)
    plt.subplots_adjust(wspace=0.02)

    country_to_color = {'Mexico': 'C2',
                        'China': 'C1',
                        'Italy': 'C0',
                        'Germany': 'C0',
                        'Ireland': 'C0',
                        'Cuba': 'C2',
                        'Japan': 'C1',
                        'Poland': 'C0',
                        'Haiti': 'C2',
                        'Russia': 'C0',
                        'Vietnam': 'C1',
                        'Hungary': 'C0',
                        'Greece': 'C0',
                        'Philippines': 'C1'
                        }

    for g_i, country in enumerate(target_countries[:nrows*2]):

        row = g_i
        col = 0
        if g_i >= nrows:
            row -= nrows
            col = 1

        plot_percent_diff_line_with_bands(axes[row][col], pro_imm_speeches_per_congress, anti_imm_speeches_per_congress, imm_speeches_per_congress, congress_range, years, 'k', label=None, fill_alpha=0.1, line_alpha=0.5, linestyle='-.')
        valid_congresses = sorted([c for c, count in imm_speeches_per_congress_by_country[country].items() if count >= 20])
        blocks = []
        for i, c in enumerate(valid_congresses):
            if i == 0:
                if c == valid_congresses[i+1]-1:
                    blocks.append([c])
            elif i == len(valid_congresses)-1:
                if c == valid_congresses[i-1]+1:
                    blocks[-1].append(c)
            elif c == valid_congresses[i+1]-1 or c == valid_congresses[i-1]+1:
                if len(blocks) == 0:
                    blocks.append([c])
                else:
                    blocks[-1].append(c)
            elif len(blocks) > 0 and len(blocks[-1]) > 0:
                blocks.append([])

        print(country, len(blocks))
        for b_i, block in enumerate(blocks):
            valid_congresses = block
            xvals = [congress_to_year(c) for c in valid_congresses]
            if b_i == 0:
                if country == 'Russia':
                    label = 'Russia / USSR'
                elif country == 'Philippines':
                    label = 'The Philippines'
                else:
                    label = country
            else:
                label = None
            xvals = [congress_to_year(c) for c in valid_congresses]
            color = country_to_color[country]
            plot_percent_diff_line_with_bands(axes[row][col], pro_imm_speeches_per_congress_by_country[country], anti_imm_speeches_per_congress_by_country[country], imm_speeches_per_congress_by_country[country], valid_congresses, xvals, color, label=label)
        add_labels_to_plot(axes[row][col], years, lower, upper, legend_loc='upper left')

    plt.savefig(os.path.join(outdir, 'tone_by_country_split.png'), bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'tone_by_country_split.pdf'), bbox_inches='tight')

    # Also make frequency plots for countries

    seaborn.set_palette(seaborn.color_palette("tab10"))
    nrows = 7
    fig, axes = plt.subplots(ncols=2, nrows=nrows, figsize=(12, 9), sharex=True, sharey=False)
    plt.subplots_adjust(hspace=0.15)
    plt.subplots_adjust(wspace=0.2)

    for g_i, country in enumerate(target_countries[:nrows*2]):

        row = g_i
        col = 0
        if g_i >= nrows:
            row -= nrows
            col = 1

        valid_congresses = sorted([c for c, count in imm_speeches_per_congress_by_country[country].items()])

        print(country, len(blocks))
        xvals = [congress_to_year(c) for c in valid_congresses]
        if country == 'Russia':
            label = 'Russia / USSR'
        elif country == 'Philippines':
            label = 'The Philippines'
        else:
            label = country
        xvals = [congress_to_year(c) for c in valid_congresses]
        color = country_to_color[country]
        plot_percent_line_with_bands(axes[row][col], imm_speeches_per_congress_by_country[country], imm_speeches_per_congress, valid_congresses, xvals, color, label=label)
        if country == 'China':
            add_labels_to_plot(axes[row][col], years, lower=-3, upper=59, legend_loc='upper right')
            axes[row][col].plot(xvals, np.ones_like(xvals)*26, 'k:', alpha=0.6)
        elif country == 'Mexico':
            add_labels_to_plot(axes[row][col], years, lower=-1.5, upper=26, legend_loc='upper left')
        else:
            add_labels_to_plot(axes[row][col], years, lower=-0.75, upper=26, legend_loc='upper left')

    plt.savefig(os.path.join(outdir, 'freqs_by_country_split.png'), bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'freqs_by_country_split.pdf'), bbox_inches='tight')


def combine_name_state_and_party(name, party, state):
    return name + ' (' + party + ', ' + state + ')'


def plot_bg_fill(ax, periods, lower, upper, party_to_color):
    for start, end, party, person, nickname in periods:
        ax.fill_between([start-0.5, end-0.5], [lower, lower], [upper, upper], color=party_to_color[party], alpha=0.1)


def plot_percent_line_with_bands(ax, numerator, denominator, keys, xvals, color, label, line_alpha=1.0, fill_alpha=0.2):
    series = np.array([numerator[k] / denominator[k] for k in keys])
    ns = np.array([denominator[k] for k in keys])
    std_pro = np.sqrt(series * (1-series) / ns)
    ax.plot(xvals, 100 * series, label=label, c=color, linewidth=1, alpha=line_alpha)
    ax.fill_between(xvals, 100 * (series - 2 * std_pro), 100 * (series + 2 * std_pro), color=color, alpha=fill_alpha)


def plot_percent_diff_line_with_bands(ax, numerator1, numerator2, denominator, keys, xvals, color, label, line_alpha=1.0, fill_alpha=0.2, linestyle='-', bands=True, linewidth=1):
    series1 = np.array([numerator1[k] / denominator[k] for k in keys])
    series2 = np.array([numerator2[k] / denominator[k] for k in keys])
    series = series1-series2
    ns = np.array([denominator[k] for k in keys])
    std1 = np.sqrt(series1 * (1-series1) / ns)
    std2 = np.sqrt(series2 * (1-series2) / ns)
    std = np.sqrt(std1**2 + std2**2)
    ax.plot(xvals, 100 * series, label=label, c=color, linewidth=linewidth, alpha=line_alpha, linestyle=linestyle)
    if bands:    
        ax.fill_between(xvals, 100 * (series - 2 * std), 100 * (series + 2 * std), color=color, alpha=fill_alpha)


def scatter_percent_diff(ax, numerator1, numerator2, denominator, keys, xvals, color, label, size=15):
    series1 = np.array([numerator1[k] / denominator[k] for k in keys])
    series2 = np.array([numerator2[k] / denominator[k] for k in keys])
    series = series1-series2
    ax.scatter(xvals, 100 * series, label=label, color=color, s=size)


def add_labels_to_plot(ax, xvals, lower=None, upper=None, xlabel=None, ylabel=None, title=None, legend_loc='upper left'):
    ax.plot(xvals, np.zeros_like(xvals), 'k--', alpha=0.4)
    ax.set_xlim(xvals[0], xvals[-1])
    if lower is not  None and upper is not None:
        ax.set_ylim(lower, upper)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if legend_loc is not None:
        ax.legend(loc=legend_loc)


if __name__ == '__main__':
    main()
