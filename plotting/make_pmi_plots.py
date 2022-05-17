import os
import re
import json
import string
import datetime as dt
from glob import glob
from optparse import OptionParser
from collections import defaultdict, Counter

import seaborn
import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sms
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import smart_open
smart_open.open = smart_open.smart_open
from gensim.models import Word2Vec

from analysis.common import get_polarization_start, get_early_analysis_range, get_modern_analysis_range
from analysis.frame_terms import get_frame_replacements, get_tagged_frame_terms
from time_periods.common import periods, year_to_congress, congress_to_year, congress_to_decade


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--tone-file', type=str, default='data/speeches/Congress/imm_speech_ids_with_tone.tsv',
                      help='.tsv file with immigration segments and tones: default=%default')
    parser.add_option('--imm-mentions-file', type=str, default='data/speeches/Congress/imm_mention_sents_parsed.jsonlist',
                      help='.tsv file with immigration mention sents parsed: default=%default')
    parser.add_option('--procedural-file', type=str, default='data/speeches/Congress/procedural_speech_ids.txt',
                      help='File with list of procdeural speeches: default=%default')
    parser.add_option('--vector-file', type=str, default='data/speeches/Congress/word_vectors/imm_tokens_all.gensim',
                      help='Gensim file with word vectors: default=%default')
    parser.add_option('--counts-dir', type=str, default='data/speeches/Congress/tagged_counts/',
                      help='Directory with tagged counts (from analysis.count_tagged_lemmas.py): default=%default')
    parser.add_option('--metaphors-dir', type=str, default='data/speeches/Congress/metaphors/',
                      help='Directory with metaphor probs (from analysis.run_metaphor_analysis.py): default=%default')
    parser.add_option('--outdir', type=str, default='plots/',
                      help='Output dir: default=%default')
    parser.add_option('--size-factor', type=int, default=60,
                      help='Multiplier for scatetrplot size: default=%default')
    parser.add_option('--query-expansion', action="store_true", default=False,
                      help='Do automatic keyword expansion using word vecotrs: default=%default')
    parser.add_option('--exp-count', type=int, default=20,
                      help='Number of terms to expand by: default=%default')


    (options, args) = parser.parse_args()

    tone_file = options.tone_file
    imm_mentions_file = options.imm_mentions_file
    procedural_file = options.procedural_file
    vector_file = options.vector_file
    counts_dir = options.counts_dir
    metaphors_dir = options.metaphors_dir
    outdir = options.outdir
    size_factor = options.size_factor
    do_query_expansion = options.query_expansion
    exp_count = options.exp_count

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    early_start, early_end = get_early_analysis_range()
    modern_start, modern_end = get_modern_analysis_range()
    polarization_start = get_polarization_start()
    first_congress = early_start
    last_congress = modern_end

    congresses = list(range(first_congress, last_congress+1))
    years = [congress_to_year(c) for c in congresses]
    xvals = (np.array(years) - years[0]) / len(congresses)
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

    print("Loading immigration token counts")
    with open(os.path.join(counts_dir, 'imm_token_counts_by_congress.json')) as f:
        imm_token_counts_by_congress = json.load(f)
    imm_token_counts_by_congress = {int(k): Counter(v) for k, v in imm_token_counts_by_congress.items()}

    with open(os.path.join(counts_dir, 'imm_token_counts_by_congress_by_party.json')) as f:
        imm_token_counts_by_congress_by_party = json.load(f)
    for party in imm_token_counts_by_congress_by_party:
        imm_token_counts_by_congress_by_party[party] = {int(k): Counter(v) for k, v in imm_token_counts_by_congress_by_party[party].items()}

    print("Loading background token counts")
    with open(os.path.join(counts_dir, 'token_counts_by_congress.json')) as f:
        token_counts_by_congress = json.load(f)
    token_counts_by_congress = {int(k): Counter(v) for k, v in token_counts_by_congress.items()}

    with open(os.path.join(counts_dir, 'token_counts_by_congress_by_party.json')) as f:
        token_counts_by_congress_by_party = json.load(f)
    for party in token_counts_by_congress_by_party:
        token_counts_by_congress_by_party[party] = {int(k): Counter(v) for k, v in token_counts_by_congress_by_party[party].items()}

    print("Loading imm sent token counts")
    with open(os.path.join(counts_dir, 'imm_sent_token_counts_by_congress.json')) as f:
        imm_sent_token_counts_by_congress = json.load(f)
    imm_sent_token_counts_by_congress = {int(k): Counter(v) for k, v in imm_sent_token_counts_by_congress.items()}

    with open(os.path.join(counts_dir, 'imm_sent_token_counts_by_congress_by_party.json')) as f:
        imm_sent_token_counts_by_congress_by_party = json.load(f)
    for party in token_counts_by_congress_by_party:
        imm_sent_token_counts_by_congress_by_party[party] = {int(k): Counter(v) for k, v in imm_sent_token_counts_by_congress_by_party[party].items()}

    with open(os.path.join(counts_dir, 'imm_sent_token_counts_by_congress_by_group.json')) as f:
        imm_sent_token_counts_by_congress_by_group = json.load(f)
    for group in imm_sent_token_counts_by_congress_by_group:
        imm_sent_token_counts_by_congress_by_group[group] = {int(k): Counter(v) for k, v in imm_sent_token_counts_by_congress_by_group[group].items()}

    tagged_frame_terms = get_tagged_frame_terms()
    if do_query_expansion:
        print("Doing query expansion")
        overall_tagged_counts = Counter()
        for congress, counter in token_counts_by_congress.items():
            overall_tagged_counts.update(counter)
        tagged_frame_terms = expand_frame_terms(tagged_frame_terms, vector_file, overall_tagged_counts, exp_count)

    # Get totals
    total_tokens_by_congress = {congress: sum(token_counts_by_congress[congress].values()) for congress in congresses}
    imm_total_tokens_by_congress = {congress: sum(imm_token_counts_by_congress[congress].values()) for congress in congresses}
    imm_sent_total_tokens_by_congress = {congress: sum(imm_sent_token_counts_by_congress[congress].values()) for congress in congresses}
    total_tokens_by_congress_by_party = {}
    imm_total_tokens_by_congress_by_party = {}
    imm_sent_total_tokens_by_congress_by_party = {}
    for party in ['D', 'R']:
        total_tokens_by_congress_by_party[party] = {congress: sum(token_counts_by_congress_by_party[party][congress].values()) for congress in congresses}
        imm_total_tokens_by_congress_by_party[party] = {congress: sum(imm_token_counts_by_congress_by_party[party][congress].values()) for congress in congresses}
        imm_sent_total_tokens_by_congress_by_party[party] = {congress: sum(imm_sent_token_counts_by_congress_by_party[party][congress].values()) for congress in congresses}

    with open(os.path.join(metaphors_dir, 'log_probs_by_congress_by_party.json')) as f:
        metaphor_log_probs_by_congress_by_party = json.load(f)

    with open(os.path.join(metaphors_dir, 'log_probs_by_congress_by_group.json')) as f:
        metaphor_log_probs_by_congress_by_group = json.load(f)

    # Make 14-frame frequency plot
    seaborn.set_palette(seaborn.color_palette("tab20"))
    nrows = 7
    fig, axes = plt.subplots(ncols=2, nrows=nrows, figsize=(14, 1.6 * nrows), sharex=True, sharey=False)
    plt.subplots_adjust(hspace=0.44)
    plt.subplots_adjust(wspace=0.12)

    alpha = 0.05
    divisor = 14 * 2

    imm_total_counts = np.array([imm_total_tokens_by_congress[c] for c in congresses])
    bg_total_counts = np.array([total_tokens_by_congress[c] for c in congresses])

    frame_order = sorted(tagged_frame_terms)
    frame_order.remove('Background')

    # get order by slope
    slopes = []
    for f_i, frame in enumerate(frame_order):
        imm_frame_freqs = count_tagged_frame_terms(congresses, imm_token_counts_by_congress, imm_total_counts, tagged_frame_terms, frame)
        bias, slope, pval, label = fit_series(imm_frame_freqs, xvals, alpha, divisor)
        # make sure significant ones come first in sort order
        if pval < alpha/divisor:
            slope *= 1e5
        slopes.append(slope)

    order = np.argsort(slopes)[::-1]
    frames_by_slope = [frame_order[i] for i in order]

    for f_i, frame in enumerate(frames_by_slope):
        col = f_i % 2
        row = f_i // 2

        imm_frame_freqs = count_tagged_frame_terms(congresses, imm_token_counts_by_congress, imm_total_counts, tagged_frame_terms, frame)
        bias, slope, pval, label = fit_series(imm_frame_freqs, xvals, alpha, divisor, scientific=True)

        bg_frame_freqs = count_tagged_frame_terms(congresses, token_counts_by_congress, bg_total_counts, tagged_frame_terms, frame)

        axes[row][col].plot(years, imm_frame_freqs, linewidth=1.5, c='C0', alpha=1.0, label='Immigration speeches')
        axes[row][col].plot(years, bg_frame_freqs, linewidth=1.5, linestyle='-.', c='C2', alpha=1.0, label='All speeches')
        axes[row][col].plot(years, [bias + slope * y for y in xvals], 'k:', alpha=0.5)
        axes[row][col].set_title(frame + ' ' + label)

        axes[row][col].set_xlim(congress_to_year(first_congress), congress_to_year(last_congress))
        if col == 0:
            axes[row][col].set_ylabel('Frequency')
            if row == 0:
                axes[row][col].legend()

    plt.savefig(os.path.join(outdir, 'manual_frame_freqs.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'manual_frame_freqs.png'), bbox_inches='tight')

    # make pmi plots
    fig, axes = plt.subplots(ncols=2, nrows=7, figsize=(14, 1.6 * 7), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.44)
    plt.subplots_adjust(wspace=0.12)


    imm_background_counts = np.array([imm_token_counts_by_congress[c]['immigration (n)'] for c in congresses])
    imm_background_freqs = imm_background_counts / imm_total_counts

    bg_background_counts = np.array([token_counts_by_congress[c]['immigration (n)'] for c in congresses])
    bg_background_freqs = bg_background_counts / bg_total_counts

    baseline_pmi = np.log(imm_background_freqs / bg_background_freqs)

    # get order by slope
    slopes = []
    for f_i, frame in enumerate(frame_order):
        imm_frame_freqs = count_tagged_frame_terms(congresses, imm_token_counts_by_congress, imm_total_counts, tagged_frame_terms, frame)
        bg_frame_freqs = count_tagged_frame_terms(congresses, token_counts_by_congress, bg_total_counts, tagged_frame_terms, frame)
        pmi = np.log(imm_frame_freqs / bg_frame_freqs) / baseline_pmi

        bias, slope, pval, label = fit_series(pmi, xvals, alpha, divisor)
        # make sure significant ones come first in sort order
        if pval < alpha/divisor:
            slope *= 1e5
        slopes.append(slope)

    order = np.argsort(slopes)[::-1]
    frames_by_slope = [frame_order[i] for i in order]

    for f_i, frame in enumerate(frames_by_slope):
        col = f_i % 2
        row = f_i // 2

        pmi_series = []

        # check limits by holding out one word at a time
        for held_out_term in tagged_frame_terms[frame]:
            imm_frame_freqs = count_tagged_frame_terms(congresses, imm_token_counts_by_congress, imm_total_counts, tagged_frame_terms, frame, held_out_term)
            bg_frame_freqs = count_tagged_frame_terms(congresses, token_counts_by_congress, bg_total_counts, tagged_frame_terms, frame, held_out_term)
            pmi = np.log(imm_frame_freqs / bg_frame_freqs) / baseline_pmi
            pmi_series.append(pmi)

        imm_frame_freqs = count_tagged_frame_terms(congresses, imm_token_counts_by_congress, imm_total_counts, tagged_frame_terms, frame)
        bg_frame_freqs = count_tagged_frame_terms(congresses, token_counts_by_congress, bg_total_counts, tagged_frame_terms, frame)
        pmi = np.log(imm_frame_freqs / bg_frame_freqs) / baseline_pmi

        bias, slope, pval, label = fit_series(pmi, xvals, alpha, divisor)
        pmi_matrix = np.vstack(pmi_series)
        pmi_min = np.min(pmi_matrix, axis=0)
        pmi_max = np.max(pmi_matrix, axis=0)
        axes[row][col].fill_between(years, pmi_min, pmi_max, color='C0', alpha=0.2)
        axes[row][col].plot(years, [bias + slope * y for y in xvals], c='C0', linestyle=':', alpha=0.5)
        axes[row][col].plot(years, np.zeros_like(years), 'k--', alpha=0.5)
        axes[row][col].plot(years, pmi, linewidth=1.5, c='C0')
        axes[row][col].set_title(frame + ' ' + label)
        axes[row][col].set_ylabel('Scaled PMI')

    plt.savefig(os.path.join(outdir, 'PMI_all.png'), bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'PMI_all.pdf'), bbox_inches='tight')

    fig, axes = plt.subplots(ncols=2, nrows=7, figsize=(14, 1.6 * 7), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.44)
    plt.subplots_adjust(wspace=0.12)

    imm_total_counts_dem = np.array([imm_total_tokens_by_congress_by_party['D'][c] for c in congresses])
    bg_total_counts_dem = np.array([total_tokens_by_congress_by_party['D'][c] for c in congresses])

    imm_background_counts_dem = np.array([imm_token_counts_by_congress_by_party['D'][c]['immigration (n)'] for c in congresses])
    imm_background_freqs_dem = imm_background_counts_dem / imm_total_counts_dem

    bg_background_counts_dem = np.array([token_counts_by_congress_by_party['D'][c]['immigration (n)'] for c in congresses])
    bg_background_freqs_dem = bg_background_counts_dem / bg_total_counts_dem

    baseline_pmi_dem = np.log(imm_background_freqs_dem / bg_background_freqs_dem)

    imm_total_counts_rep = np.array([imm_total_tokens_by_congress_by_party['R'][c] for c in congresses])
    bg_total_counts_rep = np.array([total_tokens_by_congress_by_party['R'][c] for c in congresses])

    imm_background_counts_rep = np.array([imm_token_counts_by_congress_by_party['R'][c]['immigration (n)'] for c in congresses])
    imm_background_freqs_rep = imm_background_counts_rep / imm_total_counts_rep

    bg_background_counts_rep = np.array([token_counts_by_congress_by_party['R'][c]['immigration (n)'] for c in congresses])
    bg_background_freqs_rep = bg_background_counts_rep / bg_total_counts_rep

    baseline_pmi_rep = np.log(imm_background_freqs_rep / bg_background_freqs_rep)

    # only fit the party divergence from a certain point onwards
    start = polarization_start - early_start
    end = len(congresses)+1

    slopes = []
    for f_i, frame in enumerate(frame_order):
        imm_frame_freqs_dem = count_tagged_frame_terms(congresses, imm_token_counts_by_congress_by_party['D'], imm_total_counts_dem, tagged_frame_terms, frame)
        bg_frame_freqs_dem = count_tagged_frame_terms(congresses, token_counts_by_congress_by_party['D'], bg_total_counts_dem, tagged_frame_terms, frame)
        pmi_dem = np.log(imm_frame_freqs_dem / bg_frame_freqs_dem) / baseline_pmi_dem

        imm_frame_freqs_rep = count_tagged_frame_terms(congresses, imm_token_counts_by_congress_by_party['R'], imm_total_counts_rep, tagged_frame_terms, frame)
        bg_frame_freqs_rep = count_tagged_frame_terms(congresses, token_counts_by_congress_by_party['R'], bg_total_counts_rep, tagged_frame_terms, frame)
        pmi_rep = np.log(imm_frame_freqs_rep / bg_frame_freqs_rep) / baseline_pmi_rep

        pmi_diff = pmi_rep - pmi_dem
        bias, slope, pval, label = fit_series(pmi_diff[start:end], xvals[start:end], alpha, divisor)
        slopes.append(slope)

    order = np.argsort(slopes)[::-1]
    frames_by_slope = [frame_order[i] for i in order]

    for f_i, frame in enumerate(frames_by_slope):
        col = f_i % 2
        row = f_i // 2

        pmi_series = []
        for heldout_term in tagged_frame_terms[frame]:

            imm_frame_freqs_dem = count_tagged_frame_terms(congresses, imm_token_counts_by_congress_by_party['D'], imm_total_counts_dem, tagged_frame_terms, frame, heldout_term)
            bg_frame_freqs_dem = count_tagged_frame_terms(congresses, token_counts_by_congress_by_party['D'], bg_total_counts_dem, tagged_frame_terms, frame, heldout_term)
            pmi_dem = np.log(imm_frame_freqs_dem / bg_frame_freqs_dem) / baseline_pmi_dem

            imm_frame_freqs_rep = count_tagged_frame_terms(congresses, imm_token_counts_by_congress_by_party['R'], imm_total_counts_rep, tagged_frame_terms, frame, heldout_term)
            bg_frame_freqs_rep = count_tagged_frame_terms(congresses, token_counts_by_congress_by_party['R'], bg_total_counts_rep, tagged_frame_terms, frame, heldout_term)
            pmi_rep = np.log(imm_frame_freqs_rep / bg_frame_freqs_rep) / baseline_pmi_rep

            pmi_diff = pmi_rep - pmi_dem
            pmi_series.append(pmi_diff)

        imm_frame_freqs_dem = count_tagged_frame_terms(congresses, imm_token_counts_by_congress_by_party['D'], imm_total_counts_dem, tagged_frame_terms, frame)
        bg_frame_freqs_dem = count_tagged_frame_terms(congresses, token_counts_by_congress_by_party['D'], bg_total_counts_dem, tagged_frame_terms, frame)
        pmi_dem = np.log(imm_frame_freqs_dem / bg_frame_freqs_dem) / baseline_pmi_dem

        imm_frame_freqs_rep = count_tagged_frame_terms(congresses, imm_token_counts_by_congress_by_party['R'], imm_total_counts_rep, tagged_frame_terms, frame)
        bg_frame_freqs_rep = count_tagged_frame_terms(congresses, token_counts_by_congress_by_party['R'], bg_total_counts_rep, tagged_frame_terms, frame)
        pmi_rep = np.log(imm_frame_freqs_rep / bg_frame_freqs_rep) / baseline_pmi_rep

        pmi_diff = pmi_rep - pmi_dem
        bias, slope, pval, label = fit_series(pmi_diff[start:end], xvals[start:end], alpha, divisor)

        axes[row][col].plot(years[start:end], [bias + slope * y for y in xvals[start:end]], c='C8', linestyle='--', alpha=0.5)

        pmi_matrix = np.vstack(pmi_series)
        pmi_min = np.min(pmi_matrix, axis=0)
        pmi_max = np.max(pmi_matrix, axis=0)
        axes[row][col].plot(years, pmi_diff, linewidth=1.5, c='C8')
        axes[row][col].fill_between(years, pmi_min, pmi_max, color='C8', alpha=0.2)

        axes[row][col].plot(years, np.zeros_like(years), 'k--', alpha=0.5)
        axes[row][col].set_title(frame + ' ' + label)
        if col == 0:
            axes[row][col].set_ylabel('PMI diff (R - D)')

    plt.savefig(os.path.join(outdir, 'PMI_party_diff.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'PMI_party_diff.png'), bbox_inches='tight')

    # Plot individual party lines
    fig, axes = plt.subplots(ncols=2, nrows=7, figsize=(14, 1.6 * 7), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.44)
    plt.subplots_adjust(wspace=0.12)

    # Use the same order as the difference plot
    for f_i, frame in enumerate(frames_by_slope):
        col = f_i % 2
        row = f_i // 2

        imm_frame_freqs_dem = count_tagged_frame_terms(congresses, imm_token_counts_by_congress_by_party['D'], imm_total_counts_dem, tagged_frame_terms, frame)
        bg_frame_freqs_dem = count_tagged_frame_terms(congresses, token_counts_by_congress_by_party['D'], bg_total_counts_dem, tagged_frame_terms, frame)
        pmi_dem = np.log(imm_frame_freqs_dem / bg_frame_freqs_dem) #/ baseline_pmi_dem

        bias, slope, pval, label = fit_series(pmi_dem[start:end], xvals[start:end], alpha, divisor)

        axes[row][col].plot(years, pmi_dem, linewidth=1.5, c='b', label = 'D ' + label)
        axes[row][col].plot(years[start:end], [bias + slope * y for y in xvals[start:end]], c='b', linestyle='--', alpha=0.5)

        imm_frame_freqs_rep = count_tagged_frame_terms(congresses, imm_token_counts_by_congress_by_party['R'], imm_total_counts_rep, tagged_frame_terms, frame)
        bg_frame_freqs_rep = count_tagged_frame_terms(congresses, token_counts_by_congress_by_party['R'], bg_total_counts_rep, tagged_frame_terms, frame)
        pmi_rep = np.log(imm_frame_freqs_rep / bg_frame_freqs_rep) #/ baseline_pmi_rep

        axes[row][col].plot(years, pmi_rep, linewidth=1.5, c='r', label = 'R ' + label)
        axes[row][col].plot(years[start:end], [bias + slope * y for y in xvals[start:end]], c='r', linestyle='--', alpha=0.5)

        axes[row][col].plot(years, np.zeros_like(years), 'k--', alpha=0.5)
        axes[row][col].set_title(frame)
        #axes[row][col].set_ylim(lower, upper)
        if col == 0:
            axes[row][col].set_ylabel('Scaled PMI')

    plt.savefig(os.path.join(outdir, 'PMI_by_party.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'PMI_by_party.png'), bbox_inches='tight')


    early_dem_counts = Counter()
    early_rep_counts = Counter()
    early_imm_counts = Counter()
    early_counts = Counter()
    early_dem_log_probs = []
    early_rep_log_probs = []
    for c in range(early_start, early_end+1):
        early_dem_counts.update(imm_token_counts_by_congress_by_party['D'][c])
        early_rep_counts.update(imm_token_counts_by_congress_by_party['R'][c])
        early_imm_counts.update(imm_token_counts_by_congress[c])
        early_counts.update(token_counts_by_congress[c])
        early_dem_log_probs.extend(metaphor_log_probs_by_congress_by_party['D']['combined'][str(c)])
        early_rep_log_probs.extend(metaphor_log_probs_by_congress_by_party['R']['combined'][str(c)])

    modern_dem_counts = Counter()
    modern_rep_counts = Counter()
    modern_imm_counts = Counter()
    modern_counts = Counter()
    modern_dem_log_probs = []
    modern_rep_log_probs = []
    for c in range(modern_start, modern_end+1):
        modern_dem_counts.update(imm_token_counts_by_congress_by_party['D'][c])
        modern_rep_counts.update(imm_token_counts_by_congress_by_party['R'][c])
        modern_imm_counts.update(imm_token_counts_by_congress[c])
        modern_counts.update(token_counts_by_congress[c])
        modern_dem_log_probs.extend(metaphor_log_probs_by_congress_by_party['D']['combined'][str(c)])
        modern_rep_log_probs.extend(metaphor_log_probs_by_congress_by_party['R']['combined'][str(c)])

    total_early_dem_count = sum(early_dem_counts.values())
    total_early_rep_count = sum(early_rep_counts.values())
    total_early_imm_count = sum(early_imm_counts.values())
    total_early_count = sum(early_counts.values())
    total_modern_dem_count = sum(modern_dem_counts.values())
    total_modern_rep_count = sum(modern_rep_counts.values())
    total_modern_imm_count = sum(modern_imm_counts.values())
    total_modern_count = sum(modern_counts.values())

    total_early_imm_party_sent_count = 0
    for c in range(early_start, early_end+1):
        total_early_imm_party_sent_count += sum(imm_sent_token_counts_by_congress_by_party['D'][c].values())
        total_early_imm_party_sent_count += sum(imm_sent_token_counts_by_congress_by_party['R'][c].values())

    total_modern_imm_party_sent_count = 0
    for c in range(modern_start, modern_end+1):
        total_modern_imm_party_sent_count += sum(imm_sent_token_counts_by_congress_by_party['D'][c].values())
        total_modern_imm_party_sent_count += sum(imm_sent_token_counts_by_congress_by_party['R'][c].values())

    # Make early vs modern D vs R comparison
    fig, axes = plt.subplots(ncols=2, figsize=(8, 7), sharey=False, sharex=True)
    plt.subplots_adjust(wspace=1.0)
    seaborn.set(font_scale=1.3, style='white')

    prior_count = 100

    print("Early")
    xlim = 1
    n_frames = len(frame_order)
    early_log_means = []
    early_freqs = []
    early_freq_ratios = []
    early_lowers = []
    early_uppers = []
    for f_i, frame in enumerate(frame_order):
        terms = tagged_frame_terms[frame.title()]

        heldout_log_means = []
        for heldout_term in terms:
            dem_count = 0
            rep_count = 0
            for term in terms:
                if term != heldout_term:
                    dem_count += early_dem_counts[term]
                    rep_count += early_rep_counts[term]
            dem_freq = dem_count / total_early_dem_count
            rep_freq = rep_count / total_early_rep_count
            heldout_log_means.append(np.log(rep_freq/dem_freq))
        early_lowers.append(min(heldout_log_means))
        early_uppers.append(max(heldout_log_means))

        dem_count = 0
        rep_count = 0
        imm_count = 0
        overall_count = 0
        imm_counts_by_term = Counter()
        for term in terms:
            dem_count += early_dem_counts[term]
            rep_count += early_rep_counts[term]
            imm_count += early_imm_counts[term]
            overall_count += early_counts[term]
            imm_counts_by_term[term] = early_imm_counts[term]
        print(frame, dem_count, rep_count, imm_count, dem_count/total_early_dem_count, rep_count/total_early_rep_count)
        print(imm_counts_by_term.most_common(n=5))
        alpha_1, beta_1 = get_posterior_params(dem_count, total_early_dem_count, imm_count, total_early_imm_count, prior_count)
        alpha_2, beta_2 = get_posterior_params(rep_count, total_early_rep_count, imm_count, total_early_imm_count, prior_count)
        samples = 100000
        dem_freq = dem_count / total_early_dem_count
        rep_freq = rep_count / total_early_rep_count
        overall_imm_freq = (dem_count + rep_count) / (total_early_dem_count + total_early_rep_count)
        overall_freq = overall_count / total_early_count
        rvs1 = np.random.beta(alpha_1, beta_1, size=samples)
        rvs2 = np.random.beta(alpha_2, beta_2, size=samples)
        mean = np.log(np.mean(rvs1/rvs2))
        early_log_means.append(np.log(rep_freq/dem_freq))
        early_freqs.append(overall_imm_freq)
        early_freq_ratios.append(overall_imm_freq / overall_freq)

        print(frame, rep_freq/dem_freq, np.mean(rvs2/rvs1), np.percentile(rvs1/rvs2, q=[2.5, 97.5]))

    print("\nModern")
    modern_log_means = []
    modern_freqs = []
    modern_freq_ratios = []
    modern_lowers = []
    modern_uppers = []
    for f_i, frame in enumerate(frame_order):
        terms = tagged_frame_terms[frame.title()]

        heldout_log_means = []
        for heldout_term in terms:
            dem_count = 0
            rep_count = 0
            for term in terms:
                if term != heldout_term:
                    dem_count += modern_dem_counts[term]
                    rep_count += modern_rep_counts[term]
            dem_freq = dem_count / total_modern_dem_count
            rep_freq = rep_count / total_modern_rep_count
            heldout_log_means.append(np.log(rep_freq/dem_freq))
        modern_lowers.append(min(heldout_log_means))
        modern_uppers.append(max(heldout_log_means))

        dem_count = 0
        rep_count = 0
        imm_count = 0
        overall_count = 0
        imm_counts_by_term = Counter()
        for term in terms:
            dem_count += modern_dem_counts[term]
            rep_count += modern_rep_counts[term]
            imm_count += modern_imm_counts[term]
            overall_count += modern_counts[term]
            imm_counts_by_term[term] = modern_imm_counts[term]
        print(frame, dem_count, rep_count, imm_count, dem_count / total_modern_dem_count, rep_count / total_modern_rep_count)
        print(imm_counts_by_term.most_common(n=5))
        alpha_1, beta_1 = get_posterior_params(dem_count, total_modern_dem_count, imm_count, total_modern_imm_count, prior_count)
        alpha_2, beta_2 = get_posterior_params(rep_count, total_modern_rep_count, imm_count, total_modern_imm_count, prior_count)
        samples = 100000
        dem_freq = dem_count / total_modern_dem_count
        rep_freq = rep_count / total_modern_rep_count
        overall_imm_freq = (dem_count + rep_count) / (total_modern_dem_count + total_modern_rep_count)
        overall_freq = overall_count / total_modern_count
        rvs1 = np.random.beta(alpha_1, beta_1, size=samples)
        rvs2 = np.random.beta(alpha_2, beta_2, size=samples)
        modern_log_means.append(np.log(rep_freq/dem_freq))
        modern_freqs.append(overall_imm_freq)
        modern_freq_ratios.append(overall_imm_freq / overall_freq)
        print(frame, rep_freq/dem_freq, np.mean(rvs2/rvs1), np.percentile(rvs1/rvs2, q=[2.5, 97.5]))

    combined = []
    denom = []
    for i, m in enumerate(early_log_means):
        combined.append(m * early_freqs[i] + modern_log_means[i] * modern_freqs[i])
        denom.append(early_freqs[i] + modern_freqs[i])
    order = np.argsort(np.array(combined) / np.array(denom))[::-1]

    print("\nEarly D vs R")
    log_ratio = np.log(np.mean(np.exp(early_rep_log_probs)) / np.mean(np.exp(early_dem_log_probs)))
    print(log_ratio)
    factor = (xlim - log_ratio) / xlim / 2
    color = list(np.array([0, 0, 1]) * (factor) + np.array([1, 0, 0]) * (1-factor))
    axes[0].scatter([log_ratio], 0, color=color, marker='x')

    log_ratio = np.log(np.mean(np.exp(modern_rep_log_probs)) / np.mean(np.exp(modern_dem_log_probs)))
    print(log_ratio)
    factor = (xlim - log_ratio) / xlim / 2
    color = list(np.array([0, 0, 1]) * (factor) + np.array([1, 0, 0]) * (1-factor))
    axes[1].scatter([log_ratio],  0,  color=color, marker='x')

    for i, f_i in enumerate(order):
        log_ratio = early_log_means[f_i]
        factor = (xlim - log_ratio) / xlim / 2
        color = list(np.array([0, 0, 1]) * (factor) + np.array([1, 0, 0]) * (1-factor))
        print(frame_order[f_i], log_ratio, early_freq_ratios[f_i])
        axes[0].plot([early_lowers[f_i], early_uppers[f_i]], [n_frames-i, n_frames-i], color=color, linewidth=1.5)
        axes[0].scatter([log_ratio], [n_frames-i], color=color, s=max(2, early_freq_ratios[f_i] * size_factor))

        log_ratio = modern_log_means[f_i]
        factor = (xlim - log_ratio) / xlim / 2
        color = list(np.array([0, 0, 1]) * (factor) + np.array([1, 0, 0]) * (1-factor))
        print(frame_order[f_i], log_ratio, modern_freq_ratios[f_i])
        axes[1].plot([modern_lowers[f_i], modern_uppers[f_i]], [n_frames-i, n_frames-i], color=color, linewidth=1.5)
        axes[1].scatter([log_ratio], [n_frames-i], color=color, s=max(2, modern_freq_ratios[f_i] * size_factor))

    print()
    labels = [frame_order[i] for i in order] + ['Dehumanization']
    axes[0].set_yticks(range(0,n_frames+1))
    axes[0].set_yticklabels([t.title() for t in labels[::-1]], fontsize=14)
    axes[0].set_xticks([-1, 0, 1])
    axes[0].set_xticklabels(['-1', '0', '1'], fontsize=14)
    axes[0].plot([0, 0], [-0.95, n_frames+0.95], 'k--', alpha=0.5)
    axes[0].set_ylim(-0.95, 14.95)
    axes[0].set_xlim(-1.2, 1.2)
    axes[0].set_title('(1880-1912)\n' + r'D $\Longleftrightarrow$ R')
    axes[0].set_xlabel('log(Frequency ratio)', fontsize=14)

    axes[1].set_yticks(range(0,n_frames+1))
    axes[1].set_yticklabels([t.title() for t in labels[::-1]], fontsize=14)
    axes[1].set_xticks([-1, 0, 1])
    axes[1].set_xticklabels(['-1', '0', '1'], fontsize=14)
    axes[1].plot([0, 0], [-0.95, n_frames + 0.95], 'k--', alpha=0.5)
    axes[1].set_ylim(-0.95, 14.95)
    axes[1].set_xlim(-1.2, 1.2)
    axes[1].set_title('(2001-2020)\n' + r'D $\Longleftrightarrow$ R')
    axes[1].set_xlabel('log(Frequency ratio)', fontsize=14)

    if do_query_expansion:
        plt.savefig(os.path.join(outdir, 'PMI_early_vs_modern_by_party_mod.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(outdir, 'PMI_early_vs_modern_by_party_mod.png'), bbox_inches='tight')
    else:
        plt.savefig(os.path.join(outdir, 'PMI_early_vs_modern_by_party.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(outdir, 'PMI_early_vs_modern_by_party.png'), bbox_inches='tight')

    # Do early Chinese / modern Mexican vs European
    xlim = 2.5

    early_ch_counts = Counter()
    early_eu_counts = Counter()
    early_imm_counts = Counter()
    early_counts = Counter()
    early_ch_log_probs = []
    early_eu_log_probs = []

    for c in range(early_start, early_end+1):
        early_ch_counts.update(imm_sent_token_counts_by_congress_by_group['Chinese'][c])
        early_eu_counts.update(imm_sent_token_counts_by_congress_by_group['European'][c])
        early_imm_counts.update(imm_sent_token_counts_by_congress[c])
        early_counts.update(token_counts_by_congress[c])
        early_ch_log_probs.extend(metaphor_log_probs_by_congress_by_group['Chinese']['combined'][str(c)])
        early_eu_log_probs.extend(metaphor_log_probs_by_congress_by_group['European']['combined'][str(c)])

    modern_me_counts = Counter()
    modern_eu_counts = Counter()
    modern_imm_counts = Counter()
    modern_counts = Counter()
    modern_me_log_probs = []
    modern_eu_log_probs = []
    for c in range(modern_start, modern_end+1):
        modern_me_counts.update(imm_sent_token_counts_by_congress_by_group['Mexican'][c])
        modern_eu_counts.update(imm_sent_token_counts_by_congress_by_group['European'][c])
        modern_imm_counts.update(imm_sent_token_counts_by_congress[c])
        modern_counts.update(token_counts_by_congress[c])
        modern_me_log_probs.extend(metaphor_log_probs_by_congress_by_group['Mexican']['combined'][str(c)])
        modern_eu_log_probs.extend(metaphor_log_probs_by_congress_by_group['European']['combined'][str(c)])

    total_early_ch_count = sum(early_ch_counts.values())
    total_early_eu_count = sum(early_eu_counts.values())
    total_early_imm_count = sum(early_imm_counts.values())
    total_early_count = sum(early_counts.values())
    total_modern_me_count = sum(modern_me_counts.values())
    total_modern_eu_count = sum(modern_eu_counts.values())
    total_modern_imm_count = sum(modern_imm_counts.values())
    total_modern_count = sum(modern_counts.values())

    # Make early vs modern D vs R comparison
    fig, axes = plt.subplots(ncols=2, figsize=(8, 7), sharey=False, sharex=True)
    plt.subplots_adjust(wspace=1.0)
    seaborn.set(font_scale=1.3, style='white')

    prior_count = 100

    n_frames = len(frame_order)
    early_log_means = []
    early_freqs = []
    early_freq_ratios = []
    early_lowers = []
    early_uppers = []

    for f_i, frame in enumerate(frame_order):
        terms = tagged_frame_terms[frame.title()]

        heldout_log_means = []
        for heldout_term in terms:
            ch_count = 0
            eu_count = 0
            for term in terms:
                if term != heldout_term:
                    ch_count += early_ch_counts[term]
                    eu_count += early_eu_counts[term]
            ch_freq = ch_count / total_early_ch_count
            eu_freq = eu_count / total_early_eu_count
            heldout_log_means.append(np.log(ch_freq/eu_freq))
        early_lowers.append(min(heldout_log_means))
        early_uppers.append(max(heldout_log_means))

        ch_count = 0
        eu_count = 0
        imm_count = 0
        overall_count = 0
        for term in terms:
            ch_count += early_ch_counts[term]
            eu_count += early_eu_counts[term]
            imm_count += early_imm_counts[term]
            overall_count += early_counts[term]
        print(frame, ch_count, eu_count, imm_count, ch_count/total_early_ch_count, eu_count/total_early_eu_count)
        alpha_1, beta_1 = get_posterior_params(ch_count, total_early_ch_count, imm_count, total_early_imm_count, prior_count)
        alpha_2, beta_2 = get_posterior_params(eu_count, total_early_eu_count, imm_count, total_early_imm_count, prior_count)
        samples = 100000
        ch_freq = ch_count / total_early_ch_count
        eu_freq = eu_count / total_early_eu_count
        overall_imm_freq = (ch_count + eu_count) / (total_early_ch_count + total_early_eu_count)
        overall_freq = overall_count / total_early_count
        rvs1 = np.random.beta(alpha_1, beta_1, size=samples)
        rvs2 = np.random.beta(alpha_2, beta_2, size=samples)
        early_log_means.append(np.log(ch_freq/eu_freq))
        early_freqs.append(overall_imm_freq)
        early_freq_ratios.append(overall_imm_freq / overall_freq)

    modern_log_means = []
    modern_freqs = []
    modern_freq_ratios = []
    modern_lowers = []
    modern_uppers = []
    for f_i, frame in enumerate(frame_order):
        terms = tagged_frame_terms[frame.title()]

        heldout_log_means = []
        for heldout_term in terms:
            me_count = 0
            eu_count = 0
            for term in terms:
                if term != heldout_term:
                    me_count += modern_me_counts[term]
                    eu_count += modern_eu_counts[term]
            me_freq = me_count / total_modern_me_count
            eu_freq = eu_count / total_modern_eu_count
            heldout_log_means.append(np.log(me_freq/eu_freq))
        modern_lowers.append(min(heldout_log_means))
        modern_uppers.append(max(heldout_log_means))

        me_count = 0
        eu_count = 0
        imm_count = 0
        overall_count = 0
        for term in terms:
            me_count += modern_me_counts[term]
            eu_count += modern_eu_counts[term]
            imm_count += modern_imm_counts[term]
            overall_count += modern_counts[term]
        print(frame, me_count, eu_count, imm_count, me_count / total_modern_me_count, eu_count / total_modern_eu_count)
        alpha_1, beta_1 = get_posterior_params(me_count, total_modern_me_count, imm_count, total_modern_imm_count, prior_count)
        alpha_2, beta_2 = get_posterior_params(eu_count, total_modern_eu_count, imm_count, total_modern_imm_count, prior_count)
        samples = 100000
        me_freq = me_count / total_modern_me_count
        eu_freq = eu_count / total_modern_eu_count
        overall_imm_freq = (me_count + eu_count) / (total_modern_me_count + total_modern_eu_count)
        overall_freq = overall_count / total_modern_count
        rvs1 = np.random.beta(alpha_1, beta_1, size=samples)
        rvs2 = np.random.beta(alpha_2, beta_2, size=samples)
        modern_log_means.append(np.log(me_freq/eu_freq))
        modern_freqs.append(overall_imm_freq)
        modern_freq_ratios.append(overall_imm_freq / overall_freq)
        print(frame, np.mean(eu_freq/me_freq), np.mean(rvs2/rvs1), np.percentile(rvs1/rvs2, q=[2.5, 97.5]))

    log_ratio = np.log(np.mean(np.exp(early_ch_log_probs)) / np.mean(np.exp(early_eu_log_probs)))
    print(log_ratio)
    factor = (xlim - log_ratio) / xlim / 2
    color = list(np.array([0, 0, 1]) * (factor) + np.array([1, 0, 0]) * (1-factor))
    axes[0].scatter([log_ratio], 0, color=color, marker='x')

    log_ratio = np.log(np.mean(np.exp(modern_me_log_probs)) / np.mean(np.exp(modern_eu_log_probs)))
    print(log_ratio)
    factor = (xlim - log_ratio) / xlim / 2
    color = list(np.array([0, 0, 1]) * (factor) + np.array([1, 0, 0]) * (1-factor))
    axes[1].scatter([log_ratio],  0,  color=color, marker='x')

    combined = []
    denom = []
    for i, m in enumerate(early_log_means):
        combined.append(m * early_freqs[i] + modern_log_means[i] * modern_freqs[i])
        denom.append(early_freqs[i] + modern_freqs[i])
    order = np.argsort(np.array(combined) / np.array(denom))[::-1]

    for i, f_i in enumerate(order):
        log_ratio = early_log_means[f_i]
        factor = (xlim - log_ratio) / xlim / 2
        color = list(np.array([0, 1, 0]) * (factor) + np.array([1, 0, 1]) * (1-factor))
        axes[0].plot([early_lowers[f_i], early_uppers[f_i]], [n_frames-i, n_frames-i], color=color, linewidth=1.5)
        axes[0].scatter([log_ratio], [n_frames-i], color=color, s=max(2, early_freq_ratios[f_i] * size_factor))

        log_ratio = modern_log_means[f_i]
        factor = (xlim - log_ratio) / xlim / 2
        color = list(np.array([0, 1, 0]) * (factor) + np.array([1, 0, 1]) * (1-factor))
        axes[1].plot([modern_lowers[f_i], modern_uppers[f_i]], [n_frames-i, n_frames-i], color=color, linewidth=1.5)
        axes[1].scatter([log_ratio], [n_frames-i], color=color, s=max(2, modern_freq_ratios[f_i] * size_factor))

    axes[0].set_yticks(range(0,n_frames+1))

    labels = [frame_order[i] for i in order] + ['Dehumanization']
    axes[0].set_yticks(range(0,n_frames+1))
    axes[0].set_yticklabels([t.title() for t in labels[::-1]], fontsize=14)
    axes[0].set_xticks([-2, -1, 0, 1, 2])
    axes[0].set_xticklabels(['2', '-1', '0', '1', '2'], fontsize=14)
    axes[0].plot([0, 0], [-0.95, n_frames+0.95], 'k--', alpha=0.5)
    axes[0].set_ylim(-0.95, 14.95)
    axes[0].set_xlim(-2.6, 2.6)

    axes[0].set_xlabel('log(Frequency ratio)', fontsize=14)
    axes[0].set_title('(1880-1912)\n' + r'European $\Longleftrightarrow$ Chinese')

    axes[1].set_yticks(range(0,n_frames+1))
    axes[1].set_yticks(range(0,n_frames+1))
    axes[1].set_yticklabels([t.title() for t in labels[::-1]], fontsize=14)
    axes[1].set_xticks([-2, -1, 0, 1, 2])
    axes[1].set_xticklabels(['2', '-1', '0', '1', '2'], fontsize=14)
    axes[1].plot([0, 0], [-0.95, n_frames + 0.95], 'k--', alpha=0.5)
    axes[1].set_ylim(-0.95, 14.95)
    axes[1].set_xlim(-2.6, 2.6)

    axes[1].set_title('(2001-2020)\n' + r'European $\Longleftrightarrow$ Mexican')
    axes[1].set_xlabel('log(Frequency ratio)')

    if do_query_expansion:
        plt.savefig(os.path.join(outdir, 'PMI_early_vs_modern_by_group_mod.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(outdir, 'PMI_early_vs_modern_by_group_mod.png'), bbox_inches='tight')
    else:
        plt.savefig(os.path.join(outdir, 'PMI_early_vs_modern_by_group.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(outdir, 'PMI_early_vs_modern_by_group.png'), bbox_inches='tight')


def fit_series(frame_freq, years, alpha, divisor, scientific=False):
    model = sms.OLS(endog=frame_freq, exog=sms.add_constant(years))
    results = model.fit()
    bias, slope = results.params
    pvalues = results.pvalues
    pval = pvalues[1]

    if scientific:
        label = '(slope = {:.2e}'.format(slope)
    else:
        label = '(slope = {:.2f}'.format(slope)
    if pval < alpha / divisor:
        label += '*'
    if pval < 0.0005:
        label += '; p<0.001)'
    else:
        label += '; p={:.3f})'.format(pval)

    return bias, slope, pval, label


def count_tagged_frame_terms(congresses, counts, total_counts, tagged_frame_terms, frame, held_out_term=None):
    imm_frame_counts = np.zeros(len(congresses))
    for word in tagged_frame_terms[frame]:
        if held_out_term is None or word != held_out_term:
            imm_frame_counts += np.array([counts[c][word] for c in congresses])
    imm_frame_freqs = imm_frame_counts / total_counts
    return imm_frame_freqs


def expand_frame_terms(tagged_frame_terms, vector_file, overall_tagged_counts, exp_count=20):

    vectors = Word2Vec.load(vector_file)

    # Don't add terms that refer directly to immigrants
    exclude = {'immigrant', 'alien', 'nonimmigrant', 'refugee'}

    frame_terms_untagged = {}
    frame_term_sims = defaultdict(dict)
    for frame, words in tagged_frame_terms.items():
        # split of POS tag
        word_set = set([w.split()[0] for w in words])
        frame_terms_untagged[frame] = word_set

    # first get a list of most similar terms for each frame
    terms_to_test = set()
    sorted_frames = sorted(frame_terms_untagged)
    for frame in sorted_frames:
        terms = tagged_frame_terms[frame.title()]
        terms = [t.split()[0] for t in terms]
        valid_terms = [t for t in terms if t in vectors.wv.key_to_index]
        most_similar = vectors.wv.most_similar(valid_terms, topn=500)
        for term, sim in most_similar:
            frame_term_sims[frame][term] = sim
            terms_to_test.add(term)

    # then check the similarity of those terms to each frame
    most_similar_frame_by_term = {}
    for term in sorted(terms_to_test):
        sims = []
        for frame in sorted_frames:
            terms = tagged_frame_terms[frame]
            terms = [t.split()[0] for t in terms]
            if term in set(frame_terms_untagged[frame]):
                sim = 2.0
            elif term in frame_term_sims[frame]:
                sim = frame_term_sims[frame][term]
            else:
                sim = 0.
            sims.append(sim)
            if frame == 'Crime' and term == 'crime':
                print(term, frame, sim)
        # note the most similar frame for each term
        index = int(np.argmax(sims))
        most_similar_frame_by_term[term] = sorted_frames[index]

    # then add similar terms to each frame
    augmented_frame_words = {frame: set(terms.copy()) for frame, terms in tagged_frame_terms.items()}
    for frame in sorted_frames:
        new_terms = set()
        if frame != 'Background':
            terms = tagged_frame_terms[frame]
            print(frame)
            terms = [t.split()[0] for t in terms]
            term_set = set(terms)
            valid_terms = [t for t in terms if t in vectors.wv.key_to_index]
            most_similar = vectors.wv.most_similar(valid_terms, topn=500)
            count = 0
            for term, sim in most_similar:
                # only keep those that are sufficiently frequent, new, and most similar to the current frame
                for tag in ['v', 'n', 'j']:
                    tagged_term = term + ' ({:s})'.format(tag)
                    if frame == 'Crime' and term == 'crime':
                        print(frame, term, tag, term in exclude, term in term_set, most_similar_frame_by_term[term], overall_tagged_counts[tagged_term], sim)
                    if term not in exclude and term not in term_set and most_similar_frame_by_term[term] == frame and overall_tagged_counts[tagged_term] > 500 and sim >= 0.5:
                        print(tagged_term, sim)
                        count += 1
                        augmented_frame_words[frame].add(tagged_term)
                        new_terms.add(tagged_term)
                if count >= exp_count:
                    break
            print(', '.join(sorted(new_terms)))
            print()
    return augmented_frame_words


def get_posterior_params(n_pos, n_obs, n_prior_pos, n_prior_obs, prior_count):
    alpha = n_pos + prior_count * n_prior_pos / n_prior_obs
    total = n_obs + prior_count
    beta = total - alpha
    return alpha, beta


if __name__ == '__main__':
    main()
