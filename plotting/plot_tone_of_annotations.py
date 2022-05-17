import os
import json
from glob import glob
from optparse import OptionParser
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from time_periods.common import congress_to_year, get_modern_congress_range


# Make plots of tone over time from raw annotations

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--labeled-dir', type=str, default='data/speeches/Congress/labeled/',
                      help='LAbeled dir: default=%default')
    parser.add_option('--procedural-file', type=str, default='data/speeches/Congress/procedural_speech_ids.txt',
                      help='File with list of procedural speeches: default=%default')
    parser.add_option('--metadata-dir', type=str, default='/u/scr/nlp/data/congress/metadata/',
                      help='Metadata directory: default=%default')
    parser.add_option('--outdir', type=str, default='plots/',
                      help='Issue: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    indir = options.labeled_dir
    procedural_file = options.procedural_file
    metadata_dir = options.metadata_dir
    outdir = options.outdir

    subsets = ['early', 'mid', 'modern']

    modern_start, modern_end = get_modern_congress_range()

    print("Loading annotations")
    lines_by_subset = {}
    for subset in subsets:
        infile = os.path.join(indir, subset + '_tone_all.jsonlist')
        with open(infile) as f:
            lines = f.readlines()
        lines_by_subset[subset] = [json.loads(line) for line in lines]

    print("Loading procedural ids")
    infile = procedural_file
    with open(infile) as f:
        to_exclude = f.readlines()
    to_exclude = set([n.strip() for n in to_exclude])
    len(to_exclude)

    print("Loading metadata")
    indir = metadata_dir
    party_by_id = {}
    files = sorted(glob(indir + 'metadata_*.json'))
    for infile in tqdm(files):
        with open(infile) as f:
            data = json.load(f)
        for k, v in data.items():
            party_by_id[k] = v['party']
    len(party_by_id)

    print("Aggregating data")
    n_by_congress_by_subset = defaultdict(Counter)
    pro_by_congress_by_subset = defaultdict(Counter)
    anti_by_congress_by_subset = defaultdict(Counter)
    n_by_congress_by_subset_by_party = {party: defaultdict(Counter) for party in ['R', 'D']}
    pro_by_congress_by_subset_by_party = {party: defaultdict(Counter) for party in ['R', 'D']}
    anti_by_congress_by_subset_by_party = {party: defaultdict(Counter) for party in ['R', 'D']}

    n_missing = 0

    modern_counts_by_party = defaultdict(Counter)

    for subset in subsets:
        for line in lines_by_subset[subset]:

            line_id = line['id']
            parts = line_id.split('_')
            assert len(parts) == 2
            if parts[0] not in to_exclude:
                if parts[0] not in party_by_id:
                    n_missing += 1
                else:
                    party = party_by_id[parts[0]]
                    if line_id.startswith('1'):
                        congress = int(line_id[:3])
                    else:
                        congress = int(line_id[:2])
                    label = line['label']
                    weight = line['weight']

                    n_by_congress_by_subset[subset][congress] += 1
                    if party == 'R' or party == 'D':
                        n_by_congress_by_subset_by_party[party][subset][congress] += 1                  
                        if congress >= modern_start:
                            modern_counts_by_party[party]['total'] += 1

                    if label == 'pro':
                        pro_by_congress_by_subset[subset][congress] += 1
                        if party == 'D' or party == 'R':
                            pro_by_congress_by_subset_by_party[party][subset][congress] += 1
                            if congress >= modern_start:
                                modern_counts_by_party[party][label] += 1
                    elif label == 'anti':
                        anti_by_congress_by_subset[subset][congress] += 1
                        if party == 'D' or party == 'R':
                            anti_by_congress_by_subset_by_party[party][subset][congress] += 1
                            if congress >= modern_start:
                                modern_counts_by_party[party][label] += 1

    print(modern_counts_by_party)

    congresses_by_subset_by_party = defaultdict(dict)
    stds_by_subset_by_party = defaultdict(dict)
    diffs_by_subset_by_party = defaultdict(dict)

    parties = ['D', 'R']
    for subset in subsets:
        for party in parties:
            congresses_by_subset_by_party[party][subset] = sorted([n for n, c in n_by_congress_by_subset_by_party[party][subset].items() if c > 4])
            pro_prop_by_congress = {c: pro_by_congress_by_subset_by_party[party][subset][c] / n_by_congress_by_subset_by_party[party][subset][c] for c in congresses_by_subset_by_party[party][subset]}
            anti_prop_by_congress = {c: anti_by_congress_by_subset_by_party[party][subset][c] / n_by_congress_by_subset_by_party[party][subset][c] for c in congresses_by_subset_by_party[party][subset]}
            pro_std = {c: np.sqrt((pro_prop_by_congress[c]) * (1-pro_prop_by_congress[c]) / n_by_congress_by_subset_by_party[party][subset][c]) for c in congresses_by_subset_by_party[party][subset]}
            anti_std = {c: np.sqrt((anti_prop_by_congress[c]) * (1-anti_prop_by_congress[c]) / n_by_congress_by_subset_by_party[party][subset][c]) for c in congresses_by_subset_by_party[party][subset]}
            stds_by_subset_by_party[party][subset] = {c: np.sqrt(pro_std[c] ** 2 + anti_std[c] ** 2) for c in congresses_by_subset_by_party[party][subset]}
            diffs_by_subset_by_party[party][subset] = {c: pro_prop_by_congress[c] - anti_prop_by_congress[c] for c in congresses_by_subset_by_party[party][subset]}

    #for party in parties:
    #    print(party, list(congresses_by_subset_by_party[party].keys()))

    print("Making plot")
    fig, axes = plt.subplots(nrows=2, figsize=(12,6))
    plt.subplots_adjust(hspace=0.4)

    first_year = congress_to_year(min(n_by_congress_by_subset['early']))
    last_year = congress_to_year(max(n_by_congress_by_subset['modern']))

    for subset in subsets:
        for party in parties:
            if subset == 'early':
                label = party
            else:
                label = None
            if party == 'D':
                color = 'b'
            elif party == 'R':
                color = 'r'
            else:
                color = 'green'
            congresses = congresses_by_subset_by_party[party][subset]
            years = [congress_to_year(c) for c in congresses]
            axes[0].plot(years, [100*diffs_by_subset_by_party[party][subset][c] for c in congresses], c=color, label=label)
            axes[0].fill_between(years, [100*(diffs_by_subset_by_party[party][subset][c] - 2 * stds_by_subset_by_party[party][subset][c]) for c in congresses], [100*(diffs_by_subset_by_party[party][subset][c] + 2 * stds_by_subset_by_party[party][subset][c]) for c in congresses], alpha=0.1, color=color)
            axes[1].plot(years, [n_by_congress_by_subset_by_party[party][subset][c] for c in congresses], c=color, label=label)

    axes[0].plot([first_year, last_year], [0, 0], 'k--', alpha=0.5)

    axes[0].set_title('Average tone in annotated examples (% pro - % anti)')
    axes[0].set_ylabel('% Pro - % Anti')
    axes[0].set_ylim(-100, 100)
    axes[1].set_title('Number of annotated examples per Congressional session by party')
    axes[1].set_ylabel('# annotations')
    axes[1].set_xlabel('Year')

    axes[0].legend()
    axes[1].legend(loc='upper left')

    plt.savefig(os.path.join(outdir, 'annotated_tone.png'), bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'annotated_tone.pdf'), bbox_inches='tight')


if __name__ == '__main__':
    main()
