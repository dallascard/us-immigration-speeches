import os
import json
from optparse import OptionParser
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from time_periods.common import periods, congress_to_year, congress_to_decade
from plotting.make_tone_plots import plot_bg_fill, combine_name_state_and_party, plot_percent_diff_line_with_bands, add_labels_to_plot, scatter_percent_diff

# Script for doing leave-one-out and individual speaker plots


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--tone-file', type=str, default='data/speeches/Congress/imm_speech_ids_with_tone.tsv',
                      help='.tsv file with immigration segments and tones: default=%default')
    parser.add_option('--procedural-file', type=str, default='data/speeches/Congress/procedural_speech_ids.txt',
                      help='File with list of procdeural speeches: default=%default')
    parser.add_option('--outdir', type=str, default='plots_loo/',
                      help='Output dir: default=%default')
    parser.add_option('--first-congress', type=int, default=46,
                      help='First congress: default=%default')
    parser.add_option('--last-congress', type=int, default=116,
                      help='Last congress: default=%default')
    parser.add_option('--only-keyword', action="store_true", default=False,
                      help='Only use keyword speeches: default=%default')

    (options, args) = parser.parse_args()

    tone_file = options.tone_file
    procedural_file = options.procedural_file
    outdir = options.outdir
    first_congress = options.first_congress
    last_congress = options.last_congress
    only_keyword = options.only_keyword

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    congress_range = list(range(first_congress, last_congress+1))
    years = [congress_to_year(c) for c in congress_range]
    print(years[0], years[-1])

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

    imm_speech_id_list = list(df['speech_id'].values)

    congresses = list(df['congress'].values)
    parties = list(df['party'].values)
    speakers = list(df['speaker'].values)
    states = list(df['state'].values)
    chambers = list(df['chamber'].values)
    imm_speech_id_list = list(df['speech_id'].values)

    tone_scores = np.array(df[['anti_prob_sum', 'neutral_prob_sum', 'pro_prob_sum']].values)
    tones = tone_scores.argmax(1)

    print("Identifying top speakers")
    party_by_speaker = {}
    speaker_counter = Counter()
    for i, speaker in enumerate(speakers):
        name = combine_name_state_and_party(speaker, parties[i], states[i])
        speaker_counter[name] += 1
    target_speakers = set()
    other_count = 0
    for speaker, count in speaker_counter.most_common(n=1):
        most_prolific_speaker = speaker
    print(most_prolific_speaker)
    for speaker, count in speaker_counter.most_common(n=20):
        print(speaker, count)
    for speaker, count in speaker_counter.most_common():
        if count >= 20:
            target_speakers.add(speaker)
        else:
            other_count += count
    print(len(target_speakers))
    target_speaker_set = set(target_speakers)
    other_speaker_name = 'other'
    print("other", other_count)

    imm_speeches_per_congress = Counter(congresses)
    imm_speeches_per_decade = Counter(congresses)
    pro_imm_speeches_per_congress = Counter()
    anti_imm_speeches_per_congress = Counter()
    neutral_imm_speeches_per_congress = Counter()

    imm_speeches_per_congress_by_party = defaultdict(Counter)
    pro_imm_speeches_per_congress_by_party = defaultdict(Counter)
    neutral_imm_speeches_per_congress_by_party = defaultdict(Counter)
    anti_imm_speeches_per_congress_by_party = defaultdict(Counter)

    speaker_counter = Counter()
    imm_speeches_per_congress_by_speaker = defaultdict(Counter)
    pro_imm_speeches_per_congress_by_speaker = defaultdict(Counter)
    neutral_imm_speeches_per_congress_by_speaker = defaultdict(Counter)
    anti_imm_speeches_per_congress_by_speaker = defaultdict(Counter)

    state_counter = Counter()
    imm_speeches_per_congress_by_state = defaultdict(Counter)
    pro_imm_speeches_per_congress_by_state = defaultdict(Counter)
    neutral_imm_speeches_per_congress_by_state = defaultdict(Counter)
    anti_imm_speeches_per_congress_by_state = defaultdict(Counter)

    # aggregate the tones per speech along various dimensions
    n_excluded = 0

    for i, congress in tqdm(enumerate(congresses), total=len(congresses)):
        speech_id = str(imm_speech_id_list[i])
        chamber = chambers[i]
        if speech_id in to_exclude:
            n_excluded += 1
        else:
            state = states[i]
            speaker = combine_name_state_and_party(speakers[i], parties[i], states[i])

            # get the decade as the year ending (i.e. 2011-2020 -> 2020)
            decade = congress_to_decade(congress)

            imm_speeches_per_decade[decade] += 1
            imm_speeches_per_congress_by_party[parties[i]][congress] += 1

            if speaker in target_speaker_set:
                imm_speeches_per_congress_by_speaker[speaker][congress] += 1
                speaker_counter[speaker] += 1
            else:
                imm_speeches_per_congress_by_speaker[other_speaker_name][congress] += 1
                speaker_counter[other_speaker_name] += 1
            
            state_counter[state] += 1
            imm_speeches_per_congress_by_state[state][congress] += 1

            if tones[i] == 0:
                anti_imm_speeches_per_congress[congress] += 1
                anti_imm_speeches_per_congress_by_party[parties[i]][congress] += 1
                if speaker in target_speaker_set:
                    anti_imm_speeches_per_congress_by_speaker[speaker][congress] += 1
                else:
                    anti_imm_speeches_per_congress_by_speaker[other_speaker_name][congress] += 1
                anti_imm_speeches_per_congress_by_state[state][congress] += 1

            elif tones[i] == 2:
                pro_imm_speeches_per_congress[congress] += 1
                pro_imm_speeches_per_congress_by_party[parties[i]][congress] += 1
                if speaker in target_speaker_set:
                    pro_imm_speeches_per_congress_by_speaker[speaker][congress] += 1
                else:
                    pro_imm_speeches_per_congress_by_speaker[other_speaker_name][congress] += 1
                pro_imm_speeches_per_congress_by_state[state][congress] += 1
            
            else:
                neutral_imm_speeches_per_congress[congress] += 1
                neutral_imm_speeches_per_congress_by_party[parties[i]][congress] += 1
                if speaker in target_speaker_set:
                    neutral_imm_speeches_per_congress_by_speaker[speaker][congress] += 1
                else:
                    neutral_imm_speeches_per_congress_by_speaker[other_speaker_name][congress] += 1
                neutral_imm_speeches_per_congress_by_state[state][congress] += 1

    print(n_excluded, 'excluded')
    print(sum(imm_speeches_per_congress.values()))

    for s, c in speaker_counter.most_common(n=10):
        print(s, c)

    party_to_color = {'D': 'blue', 'R': 'red'}

    # make Loo plot (by speaker)
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

    ax = 1  # by party                                     
    for speaker in tqdm(sorted(target_speaker_set)):
        plot_percent_diff_line_with_bands_loo(axes[ax],
                                              pro_imm_speeches_per_congress,
                                              anti_imm_speeches_per_congress,
                                              imm_speeches_per_congress,
                                              pro_imm_speeches_per_congress_by_speaker[speaker],
                                              anti_imm_speeches_per_congress_by_speaker[speaker],
                                              imm_speeches_per_congress_by_speaker[speaker],
                                              congress_range,
                                              years,
                                              'k',
                                              label=None,
                                              fill_alpha=0.0,
                                              line_alpha=0.2,
                                              linestyle='-',
                                              bands=False,
                                              linewidth=1)
        party = get_party(speaker)
        if party == 'R':
            color = party_to_color['R']
        elif party == 'D':
            color = party_to_color['D']
        else:
            color = None
        if color is not None:
            plot_percent_diff_line_with_bands_loo(axes[ax],
                                                pro_imm_speeches_per_congress_by_party[party],
                                                anti_imm_speeches_per_congress_by_party[party],
                                                imm_speeches_per_congress_by_party[party],
                                                pro_imm_speeches_per_congress_by_speaker[speaker],
                                                anti_imm_speeches_per_congress_by_speaker[speaker],
                                                imm_speeches_per_congress_by_speaker[speaker],
                                                congress_range,
                                                years,
                                                color,
                                                label=None,
                                                line_alpha=0.4,
                                                linestyle='-',
                                                bands=False,
                                                linewidth=1)

    plot_percent_diff_line_with_bands(axes[ax], pro_imm_speeches_per_congress, anti_imm_speeches_per_congress, imm_speeches_per_congress, congress_range, years, 'k', label='All speeches', fill_alpha=0.1, line_alpha=1.0, linewidth=0.5, linestyle='-', bands=False)
    party = 'D'
    plot_percent_diff_line_with_bands(axes[ax], pro_imm_speeches_per_congress_by_party[party], anti_imm_speeches_per_congress_by_party[party], imm_speeches_per_congress_by_party[party], congress_range, years, party_to_color[party], label='Democrat', bands=False, line_alpha=1.0, linewidth=0.5)
    party = 'R'
    plot_percent_diff_line_with_bands(axes[ax], pro_imm_speeches_per_congress_by_party[party], anti_imm_speeches_per_congress_by_party[party], imm_speeches_per_congress_by_party[party], congress_range, years, party_to_color[party], label='Republican', bands=False, line_alpha=1.0, linewidth=0.5)

    add_labels_to_plot(axes[ax], years, lower, upper, ylabel='% Pro - % Anti speeches', title='Net tone of immigration speeches in Congress by party (leaving out each speaker)')

    plt.savefig(os.path.join(outdir, 'main_tone_plot_loo_speaker.png'), bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'main_tone_plot_loo_speaker.pdf'), bbox_inches='tight')



    # plot tone by speaker
    nrows = 2
    fig, axes = plt.subplots(nrows=nrows, figsize=(12, 6.3))
    plt.subplots_adjust(hspace=0.3)

    lower = -105
    upper = 105
    for ax in range(nrows):
        plot_bg_fill(axes[ax], periods, lower, upper, party_to_color)

    min_count_per_congress = 20
    
    speaker_start_congresses = {}
    for g_ii, speaker in enumerate(target_speaker_set):
        #valid_decades = sorted([year for year, count in imm_speeches_per_decade_by_speaker[speaker].items() if count >= 20])
        valid_congresses = [year for year, count in imm_speeches_per_congress_by_speaker[speaker].items() if count >= min_count_per_congress]
        if len(valid_congresses) > 0:
            speaker_start_congresses[speaker] = min(valid_congresses)    

    speakers_to_plot = sorted(speaker_start_congresses)
    print(len(speakers_to_plot))

    for speaker in tqdm(speakers_to_plot):
        #print(speaker, speaker_start_congresses[speaker])

        party = get_party(speaker)
        state = get_state(speaker)

        #plot_percent_diff_line_with_bands(ax[0], pro_imm_speeches_per_congress, anti_imm_speeches_per_congress, imm_speeches_per_congress, congress_range, years, 'k', label=None, line_alpha=0.8, bands=False)

        valid_congresses = [year for year, count in imm_speeches_per_congress_by_speaker[speaker].items() if count >= min_count_per_congress]

        # fix an error (or duplicate name?)
        if speaker == 'Mr. LODGE (R, MA)':
            valid_congresses = [v for v in valid_congresses if v < 76]
        elif speaker == 'Mr. KENNEDY (D, MA)':
            valid_congresses = [v for v in valid_congresses if 86 < v < 112]

        xvals = [congress_to_year(d) for d in valid_congresses]

        series1 = np.array([pro_imm_speeches_per_congress_by_speaker[speaker][k] / imm_speeches_per_congress_by_speaker[speaker][k] for k in valid_congresses])
        series2 = np.array([anti_imm_speeches_per_congress_by_speaker[speaker][k] / imm_speeches_per_congress_by_speaker[speaker][k] for k in valid_congresses])
        series = series1-series2
        series_mean = np.mean(series)

        sizes = [imm_speeches_per_congress_by_speaker[speaker][k] for k in valid_congresses]

        if party == 'D':
            ax = 0
            color = party_to_color[party]
            plot_percent_diff_broken_line(axes[ax], pro_imm_speeches_per_congress_by_speaker[speaker], anti_imm_speeches_per_congress_by_speaker[speaker], imm_speeches_per_congress_by_speaker[speaker], valid_congresses, xvals, party_to_color[party], line_alpha=0.3)
            scatter_percent_diff(axes[ax], pro_imm_speeches_per_congress_by_speaker[speaker], anti_imm_speeches_per_congress_by_speaker[speaker], imm_speeches_per_congress_by_speaker[speaker], valid_congresses, xvals, party_to_color[party], size=sizes, label=None, alpha=0.3)
        elif party == 'R':
            ax = 1
            color = party_to_color[party]
            plot_percent_diff_broken_line(axes[ax], pro_imm_speeches_per_congress_by_speaker[speaker], anti_imm_speeches_per_congress_by_speaker[speaker], imm_speeches_per_congress_by_speaker[speaker], valid_congresses, xvals, party_to_color[party], line_alpha=0.3)
            scatter_percent_diff(axes[ax], pro_imm_speeches_per_congress_by_speaker[speaker], anti_imm_speeches_per_congress_by_speaker[speaker], imm_speeches_per_congress_by_speaker[speaker], valid_congresses, xvals, party_to_color[party], size=sizes, label=None, alpha=0.3)

    
    ax = 0
    add_labels_to_plot(axes[ax], years + [years[-1]+1], lower, upper, ylabel='% Pro - % Anti speeches', title='Net tone of immigration speeches by speaker (Democrats)', legend_loc=None)
    ax = 1
    add_labels_to_plot(axes[ax], years + [years[-1]+1], lower, upper, ylabel='% Pro - % Anti speeches', title='Net tone of immigration speeches by speaker (Republicans)', legend_loc=None)

    plt.savefig(os.path.join(outdir, 'tone_by_speaker.png'), bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'tone_by_speaker.pdf'), bbox_inches='tight')



    # plot tone by speaker (combined)
    nrows = 1
    fig, axis = plt.subplots(figsize=(12, 3.3))
    plt.subplots_adjust(hspace=0.3)

    ax = 0
    axes = [axis]

    lower = -105
    upper = 105
    for ax in range(nrows):
        plot_bg_fill(axes[ax], periods, lower, upper, party_to_color)

    min_count_per_congress = 20
    
    speaker_start_congresses = {}
    for g_ii, speaker in enumerate(target_speaker_set):
        #valid_decades = sorted([year for year, count in imm_speeches_per_decade_by_speaker[speaker].items() if count >= 20])
        valid_congresses = [year for year, count in imm_speeches_per_congress_by_speaker[speaker].items() if count >= min_count_per_congress]
        if len(valid_congresses) > 0:
            speaker_start_congresses[speaker] = min(valid_congresses)    

    speakers_to_plot = sorted(speaker_start_congresses)
    print(len(speakers_to_plot))

    for speaker in tqdm(speakers_to_plot):
        #print(speaker, speaker_start_congresses[speaker])

        party = get_party(speaker)
        state = get_state(speaker)

        #plot_percent_diff_line_with_bands(ax[0], pro_imm_speeches_per_congress, anti_imm_speeches_per_congress, imm_speeches_per_congress, congress_range, years, 'k', label=None, line_alpha=0.8, bands=False)

        valid_congresses = [year for year, count in imm_speeches_per_congress_by_speaker[speaker].items() if count >= min_count_per_congress]

        # fix an error (or duplicate name?)
        if speaker == 'Mr. LODGE (R, MA)':
            valid_congresses = [v for v in valid_congresses if v < 76]
        elif speaker == 'Mr. KENNEDY (D, MA)':
            valid_congresses = [v for v in valid_congresses if 86 < v < 112]

        xvals = [congress_to_year(d) for d in valid_congresses]

        series1 = np.array([pro_imm_speeches_per_congress_by_speaker[speaker][k] / imm_speeches_per_congress_by_speaker[speaker][k] for k in valid_congresses])
        series2 = np.array([anti_imm_speeches_per_congress_by_speaker[speaker][k] / imm_speeches_per_congress_by_speaker[speaker][k] for k in valid_congresses])
        series = series1-series2
        series_mean = np.mean(series)

        sizes = [imm_speeches_per_congress_by_speaker[speaker][k] for k in valid_congresses]

        if party == 'D':
            color = party_to_color[party]
            plot_percent_diff_broken_line(axes[ax], pro_imm_speeches_per_congress_by_speaker[speaker], anti_imm_speeches_per_congress_by_speaker[speaker], imm_speeches_per_congress_by_speaker[speaker], valid_congresses, xvals, party_to_color[party], line_alpha=0.3)
            scatter_percent_diff(axes[ax], pro_imm_speeches_per_congress_by_speaker[speaker], anti_imm_speeches_per_congress_by_speaker[speaker], imm_speeches_per_congress_by_speaker[speaker], valid_congresses, xvals, party_to_color[party], size=sizes, label=None, alpha=0.3)
        elif party == 'R':
            color = party_to_color[party]
            plot_percent_diff_broken_line(axes[ax], pro_imm_speeches_per_congress_by_speaker[speaker], anti_imm_speeches_per_congress_by_speaker[speaker], imm_speeches_per_congress_by_speaker[speaker], valid_congresses, xvals, party_to_color[party], line_alpha=0.3)
            scatter_percent_diff(axes[ax], pro_imm_speeches_per_congress_by_speaker[speaker], anti_imm_speeches_per_congress_by_speaker[speaker], imm_speeches_per_congress_by_speaker[speaker], valid_congresses, xvals, party_to_color[party], size=sizes, label=None, alpha=0.3)
    
    for ax in range(nrows):
        add_labels_to_plot(axes[ax], years + [years[-1]+1], lower, upper, ylabel='% Pro - % Anti speeches', title='Net tone of immigration speeches by speaker', legend_loc=None)

    plt.savefig(os.path.join(outdir, 'tone_by_speaker_combined.png'), bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'tone_by_speaker_combined.pdf'), bbox_inches='tight')


def get_party(name_party_state):
    parts = name_party_state.split('(')
    party = parts[-1].split(',')[0]
    return party

def get_state(name_party_state):
    parts = name_party_state.split('(')
    state = parts[-1].split(', ')[1][:-1]
    return state

def plot_percent_diff_broken_line(ax, numerator1, numerator2, denominator, keys, xvals, color, line_alpha=1.0, linestyle='-', linewidth=1):
    series1 = np.array([numerator1[k]  / denominator[k] for k in keys])
    series2 = np.array([numerator2[k] / denominator[k] for k in keys])
    series = series1-series2
    if len(xvals) > 1:
        for i, x in enumerate(xvals[:-1]):
            xdiff = xvals[i+1] - x
            if xdiff == 2 or xdiff == 4:
                ax.plot(xvals[i:i+2], 100 * series[i:i+2], c=color, linewidth=linewidth, alpha=line_alpha, linestyle=linestyle)

def plot_percent_diff_line_with_bands_loo(ax, numerator1, numerator2, denominator, numerator1_loo, numerator2_loo, denominator_loo, keys, xvals, color, label, line_alpha=1.0, fill_alpha=0.2, linestyle='-', bands=True, linewidth=1):
    series1 = np.array([(numerator1[k] - numerator1_loo[k]) / (denominator[k] - denominator_loo[k]) for k in keys])
    series2 = np.array([(numerator2[k] - numerator2_loo[k]) / (denominator[k] - denominator_loo[k]) for k in keys])
    series = series1-series2
    ns = np.array([denominator[k] - denominator_loo[k] for k in keys])
    std1 = np.sqrt(series1 * (1-series1) / ns)
    std2 = np.sqrt(series2 * (1-series2) / ns)
    std = np.sqrt(std1**2 + std2**2)
    ax.plot(xvals, 100 * series, label=label, c=color, linewidth=linewidth, alpha=line_alpha, linestyle=linestyle)
    if bands:
        ax.fill_between(xvals, 100 * (series - 2 * std), 100 * (series + 2 * std), color=color, alpha=fill_alpha)


if __name__ == '__main__':
    main()
