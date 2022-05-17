import os
import re
import json
import string
from glob import glob
import datetime as dt
from optparse import OptionParser
from collections import defaultdict, Counter

import seaborn
import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sms
import matplotlib.pyplot as plt


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--infile', type=str, default='data/speeches/Congress/stan/party_region_nodata_model.npz',
                      help='Model with samples estimated from stan: default=%default')
    parser.add_option('--outdir', type=str, default='plots_stan/',
                      help='Weight file: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    infile = options.infile
    outdir = options.outdir

    data = np.load(infile)
    tone_std = data['tone_std']
    year_std = data['year_std']
    overall_means = data['overall_means']
    dem_offsets = data['dem_offsets']
    rep_offsets = data['rep_offsets']
    nodata_offsets = data['nodata_offsets']
    north_offsets = data['north_offsets']
    south_offsets = data['south_offsets']
    west_offsets = data['west_offsets']
    region_std = data['region_std']
    senate_std = data['senate_std']
    senate_offsets = data['senate_offsets']

    # Make plot with party difference
    print(overall_means.shape)
    n_samples, n_years = overall_means.shape
    years = range(1873+1, 1873+n_years*2+1, 2)
    print(min(years), max(years))
    fig, axes = plt.subplots(nrows=2, figsize=(12, 5), sharey=False)

    dem_means = []
    rep_means = []
    dem_stds = []
    rep_stds = []
    diffs = []
    stds = []
    for i, year in enumerate(years):
        mean = overall_means[:, i].mean()
        std = overall_means[:, i].std()
        dem_mean = dem_offsets[:, i].mean() + mean
        dem_std = dem_offsets[:, i].std()
        rep_mean = rep_offsets[:, i].mean() + mean
        rep_std = rep_offsets[:, i].std()
        diff =  dem_mean - rep_mean
        diff_std = np.sqrt(dem_std**2 + rep_std**2)
        dem_means.append(dem_mean)
        rep_means.append(rep_mean)
        dem_std = np.sqrt(std**2 + dem_std**2)
        rep_std = np.sqrt(std**2 + rep_std**2)
        dem_stds.append(dem_std)
        rep_stds.append(rep_std)
        diffs.append(diff)
        stds.append(diff_std)
    axes[0].plot(years, dem_means, c='b', label='Democrat')
    axes[0].fill_between(years, np.array(dem_means) - 2*np.array(dem_stds), np.array(dem_means) + 2*np.array(dem_stds), color='b', alpha=0.1)
    axes[0].plot(years, rep_means, c='r', label='Republican')
    axes[0].fill_between(years, np.array(rep_means) - 2*np.array(rep_stds), np.array(rep_means) + 2*np.array(rep_stds), color='r', alpha=0.1)
    axes[1].plot(years, diffs, c='k', label='Democrat - Republican')
    axes[1].fill_between(years, np.array(diffs) - 2*np.array(stds), np.array(diffs) + 2*np.array(stds), color='k', alpha=0.1)

    for i in range(2):
        axes[i].plot(years, np.zeros_like(years), 'k:', alpha=0.5)
        axes[i].legend(loc='upper left')
        axes[i].set_ylabel('Tone')
        axes[i].set_ylim(-0.9, 1.2)
    plt.savefig(os.path.join(outdir, 'stan_party_diff.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'stan_party_diff.png'), bbox_inches='tight')

    # Make regional plot
    fig, axes = plt.subplots(nrows=3, figsize=(12, 7), sharey=True)

    north_means = []
    south_means = []
    west_means = []
    north_stds = []
    south_stds = []
    west_stds = []

    for i, year in enumerate(years):
        north_mean = north_offsets[:, i].mean()
        north_std = north_offsets[:, i].std()
        south_mean = south_offsets[:, i].mean()
        south_std = south_offsets[:, i].std()
        west_mean = west_offsets[:, i].mean()
        west_std = west_offsets[:, i].std()
        north_means.append(north_mean)
        south_means.append(south_mean)
        west_means.append(west_mean)
        north_stds.append(north_std)
        south_stds.append(south_std)
        west_stds.append(west_std)

    axes[0].plot(years, north_means, c='C0', label='Northern + Midwestern bias')
    axes[1].plot(years, south_means, c='C1', label='Southern bias')
    axes[2].plot(years, west_means, c='C2', label='Western bias')
    axes[0].fill_between(years, np.array(north_means) - 2*np.array(north_stds), np.array(north_means) + 2*np.array(north_stds), color='C0', alpha=0.1)
    axes[1].fill_between(years, np.array(south_means) - 2*np.array(south_stds), np.array(south_means) + 2*np.array(south_stds), color='C1', alpha=0.1)
    axes[2].fill_between(years, np.array(west_means) - 2*np.array(west_stds), np.array(west_means) + 2*np.array(west_stds), color='C2', alpha=0.1)
    for i in range(3):
        axes[i].plot(years, np.zeros_like(years), 'k:', alpha=0.5)
        axes[i].legend(loc='upper left')
        axes[i].set_ylabel('Tone')
        axes[i].set_ylim(-0.6, 0.6)
    plt.savefig(os.path.join(outdir, 'stan_tone_region.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'stan_tone_region.png'), bbox_inches='tight')

    # Make senate plot
    fig, ax = plt.subplots(nrows=1, figsize=(12, 2))

    senate_bias = []
    senate_stds = []

    for i, year in enumerate(years):
        senate_bias.append(senate_offsets[:, i].mean())
        senate_stds.append(senate_offsets[:, i].std())

    ax.plot(years, senate_bias, c='C4', label='Senate bias')
    ax.fill_between(years, np.array(senate_bias) - 2*np.array(senate_stds), np.array(senate_bias) + 2*np.array(senate_stds), color='C4', alpha=0.1)
    ax.plot(years, np.zeros_like(years), 'k:', alpha=0.5)
    ax.legend(loc='upper left')
    ax.set_ylabel('Tone offset')
    ax.set_ylim(-0.6, 0.6)
    plt.savefig(os.path.join(outdir, 'stan_tone_senate_offset.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'stan_tone_senate_offset.png'), bbox_inches='tight')


if __name__ == '__main__':
    main()
