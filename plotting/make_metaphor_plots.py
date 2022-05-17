import os
import json
from optparse import OptionParser
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from analysis.common import get_modern_analysis_range, get_early_analysis_range, get_polarization_start
from analysis.metaphor_terms import get_metaphor_terms
from time_periods.common import congress_to_year


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--metaphor-dir', type=str, default='data/speeches/Congress/metaphors/',
                      help='Directory with output of analysis/run_metaphor_analysis.py: default=%default')
    parser.add_option('--outdir', type=str, default='plots/',
                      help='Output directory: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    metaphor_dir = options.metaphor_dir
    outdir = options.outdir

    categories = sorted(get_metaphor_terms()) + ['combined']

    early_start, early_end = get_early_analysis_range()
    modern_start, modern_end = get_modern_analysis_range()
    polarization_start = get_polarization_start()
    first_congress = early_start
    last_congress = modern_end

    with open(os.path.join(metaphor_dir, 'log_probs_by_congress.json')) as f:
        log_probs_by_congress = json.load(f)

    with open(os.path.join(metaphor_dir, 'log_probs_by_congress_by_party.json')) as f:
        log_probs_by_congress_by_party = json.load(f)

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 9))
    plt.subplots_adjust(hspace=0.44)
    plt.subplots_adjust(wspace=0.12)

    for c_i, category in enumerate(categories):
        print(category)
        col = c_i % 2
        row = c_i // 2

        congresses_early = range(first_congress, polarization_start+1)
        years_early = [congress_to_year(c) for c in congresses_early]
        mean = np.array([np.mean(log_probs_by_congress[category][str(y)]) for y in congresses_early])
        congresses_modern = range(polarization_start, last_congress+1)
        years_modern = [congress_to_year(c) for c in congresses_modern]
        mean_dem = np.array([np.mean(log_probs_by_congress_by_party['D'][category][str(y)]) for y in congresses_modern])
        mean_rep = np.array([np.mean(log_probs_by_congress_by_party['R'][category][str(y)]) for y in congresses_modern])
        axes[row][col].plot(years_early, mean, c='k', alpha=0.8, label='All')
        axes[row][col].plot(years_modern, mean_dem, c='b', alpha=0.8, label='Democrats')
        axes[row][col].plot(years_modern, mean_rep, c='red', alpha=0.8, label='Republicans')
        axes[row][col].set_title(category.title())
        if c_i == 0:
            axes[row][col].legend()
        if col == 0:
            axes[row][col].set_ylabel('log(p(word | context))')


    axes[3][1].axis('off')

    plt.savefig(os.path.join(outdir, 'metaphor_log_plots.png'), bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'metaphor_log_plots.pdf'), bbox_inches='tight')



    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 9))
    plt.subplots_adjust(hspace=0.44)
    plt.subplots_adjust(wspace=0.2)

    for c_i, category in enumerate(categories):
        print(category)
        col = c_i % 2
        row = c_i // 2

        congresses_early = range(first_congress, polarization_start+1)
        years_early = [congress_to_year(c) for c in congresses_early]
        mean = np.array([np.mean(np.exp(log_probs_by_congress[category][str(y)])) for y in congresses_early])
        congresses_modern = range(polarization_start, last_congress+1)
        years_modern = [congress_to_year(c) for c in congresses_modern]
        mean_dem = np.array([np.mean(np.exp(log_probs_by_congress_by_party['D'][category][str(y)])) for y in congresses_modern])
        mean_rep = np.array([np.mean(np.exp(log_probs_by_congress_by_party['R'][category][str(y)])) for y in congresses_modern])
        axes[row][col].plot(years_early, mean, c='k', alpha=0.8, label='All')
        axes[row][col].plot(years_modern, mean_dem, c='b', alpha=0.8, label='Democrats')
        axes[row][col].plot(years_modern, mean_rep, c='red', alpha=0.8, label='Republicans')
        axes[row][col].set_title(category.title())
        if c_i == 0:
            axes[row][col].legend()
        if col == 0:
            axes[row][col].set_ylabel('p(word | context)')

    axes[3][1].axis('off')

    plt.savefig(os.path.join(outdir, 'metaphor_plots.png'), bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'metaphor_plots.pdf'), bbox_inches='tight')





if __name__ == '__main__':
    main()
