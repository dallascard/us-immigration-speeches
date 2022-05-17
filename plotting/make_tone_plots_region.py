import os
import json
from optparse import OptionParser
from collections import defaultdict, Counter

import seaborn
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from time_periods.common import congress_to_year, congress_to_decade
from plotting.make_tone_plots import combine_name_state_and_party, plot_percent_diff_line_with_bands, add_labels_to_plot

# Make the tone plot by region (in supplementary)

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--tone-file', type=str, default='data/speeches/Congress/imm_speech_ids_with_tone.tsv',
                      help='.tsv file with immigration segments and tones: default=%default')
    parser.add_option('--pres-file', type=str, default='data/speeches/Congress/pres_imm_segments_with_tone.jsonlist',
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
    #parser.add_option('--latin-america', action="store_true", default=False,
    #                  help='Use Latin America rathr than Mexico: default=%default')

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
    #latin_america = options.latin_america

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    congress_range = list(range(first_congress, last_congress+1))
    years = [congress_to_year(c) for c in congress_range]
    print(years[0], years[-1])

    print("Loading country mentions")
    with open(os.path.join(country_dir, 'imm_region_speech_ids_by_regionality_or_region_mentions.json')) as f:
        speech_ids_by_region = json.load(f)
    with open(os.path.join(country_dir, 'imm_region_counts_regionality_mentions.json')) as f:
        regionality_mentions = json.load(f)

    # exclude India and England because their numbers are not reliable
    regions = sorted(regionality_mentions)
    print(len(regions))
    counts = [regionality_mentions[g] for g in regions]
    order = np.argsort(counts)[::-1]
    target_regions = [regions[i] for i in  order[:3]]
    target_regions_set = set(target_regions)
    print("Target regionalities:")
    print(target_regions)

    fb_df = pd.read_csv(os.path.join('data', 'foreign_born_transpose.tsv'), header=0, index_col=None, sep='\t')

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
    chambers = list(df['chamber'].values)
    imm_speech_id_list = list(df['speech_id'].values)

    tone_scores = np.array(df[['anti_prob_sum', 'neutral_prob_sum', 'pro_prob_sum']].values)
    tones = tone_scores.argmax(1)

    tone_sums = tone_scores.sum(1)
    tone_probs = tone_scores / tone_sums.reshape((len(tone_scores), 1))
    tone_max_probs = tone_probs.max(1)    
    print("Tone probs")
    print(np.mean(tone_max_probs), np.median(tone_max_probs), np.mean(tone_max_probs > 0.99))

    imm_speeches_per_congress = Counter()
    imm_speeches_per_decade = Counter()
    pro_imm_speeches_per_congress = Counter()
    anti_imm_speeches_per_congress = Counter()
    neutral_imm_speeches_per_congress = Counter()

    imm_speeches_per_congress_by_region = defaultdict(Counter)
    pro_imm_speeches_per_congress_by_region = defaultdict(Counter)
    neutral_imm_speeches_per_congress_by_region = defaultdict(Counter)
    anti_imm_speeches_per_congress_by_region = defaultdict(Counter)

    # aggregate the tones per speech along various dimensions
    n_excluded = 0
    for i, congress in tqdm(enumerate(congresses), total=len(congresses)):
        speech_id = str(imm_speech_id_list[i])
        chamber = chambers[i]
        if speech_id in to_exclude:
            n_excluded += 1
        else:
            speaker = combine_name_state_and_party(speakers[i], parties[i], states[i])

            # get the decade as the year ending (i.e. 2011-2020 -> 2020)
            decade = congress_to_decade(congress)
            
            regions_mentioned = set()

            imm_speeches_per_congress[congress] += 1

            # determine the countries mentioned in this speech
            for region in target_regions_set:
                if speech_id in speech_ids_by_region[region]:
                    regions_mentioned.add(region)
                    imm_speeches_per_congress_by_region[region][congress] += 1
                    #if parties[i] == 'D' or parties[i] == 'R':
                    #    imm_speeches_per_congress_by_country_and_party[parties[i]][country][decade] += 1

            if tones[i] == 0:
                anti_imm_speeches_per_congress[congress] += 1                
                #anti_imm_speeches_per_congress_by_region[state_to_region(states[i])][congress] += 1
                for region in regions_mentioned:
                    anti_imm_speeches_per_congress_by_region[region][congress] += 1
                    #if parties[i] == 'D' or parties[i] == 'R':
                    #    anti_imm_speeches_per_congress_by_country_and_party[parties[i]][country][decade] += 1

            elif tones[i] == 2:
                pro_imm_speeches_per_congress[congress] += 1
                #pro_imm_speeches_per_congress_by_region[state_to_region(states[i])][congress] += 1
                for region in regions_mentioned:
                    pro_imm_speeches_per_congress_by_region[region][congress] += 1
                    #if parties[i] == 'D' or parties[i] == 'R':
                    #    pro_imm_speeches_per_congress_by_country_and_party[parties[i]][country][decade] += 1

            else:
                neutral_imm_speeches_per_congress[congress] += 1
                #neutral_imm_speeches_per_congress_by_region[state_to_region(states[i])][congress] += 1
                for region in regions_mentioned:
                    neutral_imm_speeches_per_congress_by_region[region][congress] += 1
                    #if parties[i] == 'D' or parties[i] == 'R':
                    #    neutral_imm_speeches_per_congress_by_country_and_party[parties[i]][country][decade] += 1


    # make plot with three regions (FB share of total pop)

    region_to_color = {'Latin America': 'C2',
                      'Asia': 'C1',
                      'Europe': 'C0'
                      }

    lower = -105
    upper = 105

    seaborn.set_palette(seaborn.color_palette("tab10"))
    #fig, axes = plt.subplots(figsize=(12, 3))
    fig, axes = plt.subplots(nrows=2, figsize=(12, 5), gridspec_kw={'height_ratios': [1.6, 1]}, sharex=True)
    plt.subplots_adjust(hspace=0)

    ax = axes[0]
    plot_percent_diff_line_with_bands(ax, pro_imm_speeches_per_congress, anti_imm_speeches_per_congress, imm_speeches_per_congress, congress_range, years, 'k', label='All speeches', fill_alpha=0.1, line_alpha=0.5, linestyle='-.')

    to_plot = ['Europe', 'Asia', 'Latin America']
    for g_i, region in enumerate(to_plot):
        valid_congresses = sorted([year for year, count in imm_speeches_per_congress_by_region[region].items() if count >= 20])
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

        print(region, len(blocks))
        for b_i, block in enumerate(blocks):
            valid_congresses = block
            xvals = [congress_to_year(c) for c in valid_congresses]
            if b_i == 0:
                label = region
            else:
                label = None
            plot_percent_diff_line_with_bands(ax, pro_imm_speeches_per_congress_by_region[region], anti_imm_speeches_per_congress_by_region[region], imm_speeches_per_congress_by_region[region], valid_congresses, xvals, region_to_color[region], label=label)
            #scatter_percent_diff(ax, pro_imm_speeches_per_congress_by_country[country], anti_imm_speeches_per_congress_by_country[country], imm_speeches_per_congress_by_country[country], valid_congresses, xvals,  'C' + str(g_i), label=None)

    ax.set_yticks([-80, -40, 0, 40, 80])
    add_labels_to_plot(ax, years, lower, upper, ylabel='% Pro - % Anti speeches')


    # get population numbers from 1880 onwards
    years_fb = fb_df['Year'].values[3:]
    total_pop = fb_df['Total Population'].values[3:]
    total_fb = fb_df['Total Foreign Born'].values[3:]
    europe_fb = fb_df['Europe'].values[3:]
    asia_fb = fb_df['Asia'].values[3:]
    latainamerica_fb = fb_df['Latin America'].values[3:]

    # interpolate missing values
    europe_fb[6] = (europe_fb[5] + europe_fb[7]) / 2
    latainamerica_fb[6:8] = (latainamerica_fb[5] + latainamerica_fb[8]) / 2
    asia_fb[6] = (asia_fb[5] + asia_fb[7]) / 2


    ax = axes[1]
    ax.plot(years_fb, 100*europe_fb/total_fb, c='C0')
    ax.fill_between(years_fb, np.zeros_like(years_fb), 100*europe_fb/total_fb, color=region_to_color['Europe'], alpha=0.6, label='Europe')
    baseline = 100*europe_fb/total_fb
    ax.plot(years_fb, baseline + 100*asia_fb/total_fb, c='C1')
    ax.fill_between(years_fb, baseline, baseline + 100*asia_fb/total_fb, color=region_to_color['Asia'], alpha=0.6, label='Asia')
    baseline += 100*asia_fb/total_fb
    ax.plot(years_fb, baseline + 100*latainamerica_fb/total_fb, c='C2')
    ax.fill_between(years_fb, baseline, baseline + 100*latainamerica_fb/total_fb, color=region_to_color['Latin America'], alpha=0.6, label='Latin America')
    baseline += 100*latainamerica_fb/total_fb
    #ax.plot(years_fb, baseline + 100*latainamerica_fb/total_fb, c='C2')    
    ax.fill_between(years_fb, baseline, 100, color='k', alpha=0.2, label='Other')
        
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='lower left')        
    ax.set_ylim(0, 100)
    ax.set_ylabel('% of US pop.\nborn in region')

    plt.savefig(os.path.join(outdir, 'tone_by_region_four_fb.png'), bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'tone_by_region_four_fb.pdf'), bbox_inches='tight')




if __name__ == '__main__':
    main()
