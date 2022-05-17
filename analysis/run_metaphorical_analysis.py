import os
import json
from optparse import OptionParser
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm
import matplotlib.pyplot as plt

from analysis.metaphor_terms import get_metaphor_terms
from analysis.common import get_early_analysis_range, get_modern_analysis_range, get_polarization_start


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--emb-dir', type=str, default='data/speeches/Congress/contextual-embeddings/bert-base-uncased/',
                      help='Embeddings dir: default=%default')
    parser.add_option('--tone-file', type=str, default='data/speeches/Congress/imm_speech_ids_with_tone.tsv',
                      help='.tsv file with immigration segments and tones: default=%default')
    parser.add_option('--procedural-file', type=str, default='data/speeches/Congress/procedural_speech_ids.txt',
                      help='File with list of procdeural speeches: default=%default')
    parser.add_option('--imm-mention-file', type=str, default='data/speeches/Congress/imm_mention_sents_parsed.jsonlist',
                      help='File with parsed sentences containing immigrant mentions (from analysis.identify_immigrant_mentions.py): default=%default')
    parser.add_option('--imm-groups-file', type=str, default='data/speeches/Congress/tagged_counts/imm_mention_sent_indices_by_group.jsonlist',
                      help='Sent indices of mention sentences with group mentions (from analysis.identify_group_mentions.py): default=%default')
    parser.add_option('--outdir', type=str, default='data/speeches/Congress/metaphors/',
                      help='Output directory: default=%default')
    parser.add_option('--samples', type=int, default=20000,
                      help='Samples for pemutation tests: default=%default')
    parser.add_option('--seed', type=int, default=35873,
                      help='Random seed: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    emb_dir = options.emb_dir
    tone_file = options.tone_file
    procedural_file = options.procedural_file
    imm_mention_file = options.imm_mention_file
    imm_groups_file = options.imm_groups_file
    outdir = options.outdir
    samples = options.samples
    seed = options.seed
    np.random.seed(seed)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    metaphor_terms = get_metaphor_terms()

    print("Loading speech data")
    df_new = pd.read_csv(tone_file, header=0, index_col=0, sep='\t')

    # Load speeches to exclude
    print("Loading procedural speech ids")
    with open(procedural_file) as f:
        to_exclude = f.readlines()
    to_exclude = set([s.strip() for s in to_exclude])
    print(len(to_exclude))

    with open(imm_groups_file) as f:
        imm_sent_indices_by_group = json.load(f)
    groups = sorted(imm_sent_indices_by_group)
    for group in groups:
        imm_sent_indices_by_group[group] = set(imm_sent_indices_by_group[group])

    imm_speech_id_list = [str(i) for i in df_new['speech_id'].values]

    id_to_party = dict(zip(imm_speech_id_list, df_new['party'].values))
    id_to_congress = dict(zip(imm_speech_id_list, df_new['congress'].values))

    with open(imm_mention_file) as f:
        imm_sent_lines = f.readlines()
    imm_sent_lines = [json.loads(line) for line in imm_sent_lines]
    print("Loaded {:d} sentences".format(len(imm_sent_lines)))

    imm_sent_lines_by_ids = {str(line['id']) + '_' + str(line['sent_index']): line for line in imm_sent_lines}

    print("Loading probs")
    metaphor_log_prob_vectors = {}
    categories = sorted(metaphor_terms)
    categories.append('combined')
    for metaphor in categories:
        probs_file = os.path.join(emb_dir, 'metaphor_log_probs_' + metaphor + '.npz')
        log_probs = np.load(probs_file)['log_probs']
        metaphor_log_prob_vectors[metaphor] = log_probs

    with open(os.path.join(emb_dir, 'masked_terms.json')) as f:
        all_words = json.load(f)

    with open(os.path.join(emb_dir, 'speech_ids.json')) as f:
        all_speech_ids = json.load(f)

    with open(os.path.join(emb_dir, 'sent_indices.json')) as f:
        all_sent_indices = json.load(f)

    print("Loaded {:d} mention probs".format(len(all_words)))

    early_start, early_end = get_early_analysis_range()
    modern_start, modern_end = get_modern_analysis_range()
    polarization_start = get_polarization_start()
    first_congress = early_start
    last_congress = modern_end

    congresses = []
    parties = []
    dems = []
    reps = []
    selected_words = []
    selected_speech_ids = []
    selected_sent_indices = []
    selected_log_probs = defaultdict(list)
    log_probs_by_congress = {}
    log_probs_by_congress_dem = {}
    log_probs_by_congress_rep = {}
    log_probs_by_congress_chinese = {}
    log_probs_by_congress_european = {}
    log_probs_by_congress_mexican = {}
    log_probs_by_congress_hispanic = {}
    for category in categories:
        log_probs_by_congress[category] = defaultdict(list)
        log_probs_by_congress_dem[category] = defaultdict(list)
        log_probs_by_congress_rep[category] = defaultdict(list)
        log_probs_by_congress_chinese[category] = defaultdict(list)
        log_probs_by_congress_european[category] = defaultdict(list)
        log_probs_by_congress_mexican[category] = defaultdict(list)
        log_probs_by_congress_hispanic[category] = defaultdict(list)

    word_counter = Counter()

    # Process each embedded mention
    print("Collecting probabilities")
    for i, speech_id in tqdm(enumerate(all_speech_ids), total=len(all_speech_ids)):
        sent_index = all_sent_indices[i]
        congress = int(id_to_congress[speech_id])
        party = id_to_party[speech_id]
        word = all_words[i]
        if first_congress <= congress <= last_congress and str(speech_id) not in to_exclude:
            word_counter[word] += 1
            congresses.append(congress)
            congress_str = str(congress)
            parties.append(party)
            selected_words.append(word)
            selected_speech_ids.append(speech_id)
            selected_sent_indices.append(sent_index)
            for category in categories:
                log_prob = float(metaphor_log_prob_vectors[category][i])
                selected_log_probs[category].append(log_prob)
                log_probs_by_congress[category][congress_str].append(log_prob)
                if party == 'D':
                    dems.append(1)
                    reps.append(0)
                    log_probs_by_congress_dem[category][congress_str].append(log_prob)
                elif party == 'R':
                    dems.append(0)
                    reps.append(1)
                    log_probs_by_congress_rep[category][congress_str].append(log_prob)
                else:
                    dems.append(0)
                    reps.append(0)
                sent_id = str(speech_id) + '_' + str(sent_index)
                if sent_id in imm_sent_indices_by_group['Chinese']:
                    log_probs_by_congress_chinese[category][congress_str].append(log_prob)
                if sent_id in imm_sent_indices_by_group['European']:
                    log_probs_by_congress_european[category][congress_str].append(log_prob)
                if sent_id in imm_sent_indices_by_group['Mexican']:
                    log_probs_by_congress_mexican[category][congress_str].append(log_prob)
                if sent_id in imm_sent_indices_by_group['Hispanic']:
                    log_probs_by_congress_hispanic[category][congress_str].append(log_prob)

    for category in categories:
        outfile = os.path.join(outdir, 'examples_' + category + '.txt')
        order = np.argsort(selected_log_probs[category])[::-1]
        with open(outfile, 'w') as f:
            for i in order[:25]:
                speech_id = selected_speech_ids[i]
                sent_index = selected_sent_indices[i]
                line = imm_sent_lines_by_ids[str(speech_id) + '_' + str(sent_index)]
                sent = line['simplified']
                f.write('{:s}\t{:.4f}\t{:s}\t{:d}\t{:s}\t'.format(speech_id, selected_log_probs[category][i], selected_words[i], congresses[i], parties[i]))
                f.write(sent + '\n')
                f.write('\n')

    for category in categories:
        outfile = os.path.join(outdir, 'modern_examples_' + category + '.txt')
        order = np.argsort(selected_log_probs[category])[::-1]
        count = 0
        kk = 0
        with open(outfile, 'w') as f:            
            while count < 25:
                i = order[kk]
                if congresses[i] >= modern_start:
                    speech_id = selected_speech_ids[i]
                    sent_index = selected_sent_indices[i]
                    line = imm_sent_lines_by_ids[str(speech_id) + '_' + str(sent_index)]
                    sent = line['simplified']
                    f.write('{:s}\t{:.4f}\t{:s}\t{:d}\t{:s}\t'.format(speech_id, selected_log_probs[category][i], selected_words[i], congresses[i], parties[i]))
                    f.write(sent + '\n')
                    f.write('\n')
                    count += 1
                kk += 1

    print("Saving probs")
    with open(os.path.join(outdir, 'log_probs_by_congress.json'), 'w') as f:
        json.dump(log_probs_by_congress, f, indent=2)

    with open(os.path.join(outdir, 'log_probs_by_congress_by_party.json'), 'w') as f:
        json.dump({'D': log_probs_by_congress_dem, 'R': log_probs_by_congress_rep}, f, indent=2)

    with open(os.path.join(outdir, 'log_probs_by_congress_by_group.json'), 'w') as f:
        json.dump({'Chinese': log_probs_by_congress_chinese,
                   'European': log_probs_by_congress_european,
                   'Mexican': log_probs_by_congress_mexican,
                   'Hispanic': log_probs_by_congress_hispanic
                   }, f, indent=2)

    # Do regressions
    columns = ['metaphor', 'overall_slope', 'overall_slope_pval', 'overall_slope_unlogged', 'overall_slope_unlogged_pval', 'party_split', 'party_split_pval', 'party_split_unlogged', 'party_split_unlogged_pval']
    rows = []

    print("Running analyses")
    for metaphor in categories:
        print(metaphor)

        model = sm.OLS(endog=selected_log_probs[metaphor], exog=sm.add_constant([c-first_congress for c in congresses]))
        fit = model.fit()
        slope = fit.params[1]
        slope_pval = fit.pvalues[1]

        model = sm.OLS(endog=np.exp(selected_log_probs[metaphor]), exog=sm.add_constant([c-43 for c in congresses]))
        fit = model.fit()
        slope_unlogged = fit.params[1]
        slope_unlogged_pval = fit.pvalues[1]

        modern_indices = [i for i, c in enumerate(congresses) if c >= polarization_start and (parties[i] == 'R' or parties[i] == 'D')]
        congresses_modern = [congresses[i]-polarization_start for i in modern_indices]
        dems_modern = [dems[i] for i in modern_indices]
        log_probs_modern = [selected_log_probs[metaphor][i] for i in modern_indices]

        temp_df = pd.DataFrame()
        temp_df['congress'] = congresses_modern
        temp_df['D_x_congress'] = np.array(congresses_modern) * np.array(dems_modern)

        model = sm.OLS(endog=log_probs_modern, exog=sm.add_constant(temp_df))
        fit = model.fit()
        slope_diff = fit.params[2]
        diff_pval = fit.pvalues[2]

        model = sm.OLS(endog=np.exp(log_probs_modern), exog=sm.add_constant(temp_df))
        fit = model.fit()
        slope_diff_unlogged = fit.params[2]
        diff_unlogged_pval = fit.pvalues[2]

        row = [metaphor, slope, slope_pval, slope_unlogged, slope_unlogged_pval, slope_diff, diff_pval, slope_diff_unlogged, diff_unlogged_pval]
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    print(df)

    outfile = os.path.join(outdir, 'metaphor_regressions.csv')
    df.to_csv(outfile)

    # Do period group comparisons
    columns = ['metaphor', 'comparison', 'period', 'diff', 'pvalue', 'diff_logged', 'pvalue_logged', 'log_ratio']
    rows = []

    for metaphor in categories:
        print(metaphor)
        dem_log_probs = []
        rep_log_probs = []
        chinese_log_probs = []
        european_log_probs = []
        for congress in range(early_start, early_end+1):
            congress_str = str(congress)
            dem_log_probs.extend(log_probs_by_congress_dem[metaphor][congress_str])
            rep_log_probs.extend(log_probs_by_congress_rep[metaphor][congress_str])
            chinese_log_probs.extend(log_probs_by_congress_chinese[metaphor][congress_str])
            european_log_probs.extend(log_probs_by_congress_european[metaphor][congress_str])

        party_diff, party_diff_pvalue = do_permutation_test(np.exp(rep_log_probs), np.exp(dem_log_probs), samples)
        group_diff, group_diff_pvalue = do_permutation_test(np.exp(chinese_log_probs), np.exp(european_log_probs), samples)

        party_diff_logged, party_diff_pvalue_logged = do_permutation_test(rep_log_probs, dem_log_probs, samples)
        group_diff_logged, group_diff_pvalue_logged = do_permutation_test(chinese_log_probs, european_log_probs, samples)

        party_log_ratio = np.log(np.mean(np.exp(rep_log_probs)) / np.mean(np.exp(dem_log_probs)))
        group_log_ratio = np.log(np.mean(np.exp(chinese_log_probs)) / np.mean(np.exp(european_log_probs)))

        row = [metaphor, 'R - D', str(early_start) + ' - ' + str(early_end), party_diff, party_diff_pvalue, party_diff_logged, party_diff_pvalue_logged, party_log_ratio]
        rows.append(row)
        row = [metaphor, 'Chinese - European', str(early_start) + ' - ' + str(early_end), group_diff, group_diff_pvalue, group_diff_logged, group_diff_pvalue_logged, group_log_ratio]
        rows.append(row)

        dem_log_probs = []
        rep_log_probs = []
        mexican_log_probs = []
        hispanic_log_probs = []
        european_log_probs = []
        dem_log_control_probs = []
        rep_log_control_probs = []
        mexican_log_control_probs = []
        hispanic_log_control_probs = []
        european_log_control_probs = []
        for congress in range(modern_start, modern_end+1):
            congress_str = str(congress)
            dem_log_probs.extend(log_probs_by_congress_dem[metaphor][congress_str])
            rep_log_probs.extend(log_probs_by_congress_rep[metaphor][congress_str])
            mexican_log_probs.extend(log_probs_by_congress_mexican[metaphor][congress_str])
            hispanic_log_probs.extend(log_probs_by_congress_hispanic[metaphor][congress_str])
            european_log_probs.extend(log_probs_by_congress_european[metaphor][congress_str])
            dem_log_control_probs.extend(log_probs_by_congress_dem['random'][congress_str])
            rep_log_control_probs.extend(log_probs_by_congress_rep['random'][congress_str])
            mexican_log_control_probs.extend(log_probs_by_congress_mexican['random'][congress_str])
            hispanic_log_control_probs.extend(log_probs_by_congress_hispanic['random'][congress_str])
            european_log_control_probs.extend(log_probs_by_congress_european['random'][congress_str])

        party_diff, party_diff_pvalue = do_permutation_test(np.exp(rep_log_probs), np.exp(dem_log_probs), samples)
        group_diff, group_diff_pvalue = do_permutation_test(np.exp(mexican_log_probs), np.exp(european_log_probs), samples)
        hispanic_diff, hispanic_diff_pvalue = do_permutation_test(np.exp(hispanic_log_probs), np.exp(european_log_probs), samples)

        party_diff_logged, party_diff_pvalue_logged = do_permutation_test(rep_log_probs, dem_log_probs, samples)
        group_diff_logged, group_diff_pvalue_logged = do_permutation_test(mexican_log_probs, european_log_probs, samples)
        hispanic_diff_logged, hispanic_diff_pvalue_logged = do_permutation_test(hispanic_log_probs, european_log_probs, samples)

        party_log_ratio = np.log(np.mean(np.exp(rep_log_probs)) / np.mean(np.exp(dem_log_probs)))
        group_log_ratio = np.log(np.mean(np.exp(mexican_log_probs)) / np.mean(np.exp(european_log_probs)))
        hispanic_log_ratio = np.log(np.mean(np.exp(hispanic_log_probs)) / np.mean(np.exp(european_log_probs)))
        #party_log_ratio = np.mean(rep_log_probs) - np.mean(dem_log_probs)
        #group_log_ratio = np.mean(mexican_log_probs) - np.mean(european_log_probs)

        row = [metaphor, 'R - D', str(modern_start) + ' - ' + str(modern_end), party_diff, party_diff_pvalue, party_diff_logged, party_diff_pvalue_logged, party_log_ratio]
        rows.append(row)
        row = [metaphor, 'Mexican - European', str(modern_start) + ' - ' + str(modern_end), group_diff, group_diff_pvalue, group_diff_logged, group_diff_pvalue_logged, group_log_ratio]
        rows.append(row)
        row = [metaphor, 'Hispanic - European', str(modern_start) + ' - ' + str(modern_end), hispanic_diff, hispanic_diff_pvalue, hispanic_diff_logged, hispanic_diff_pvalue_logged, hispanic_log_ratio]
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    print(df)

    outfile = os.path.join(outdir, 'metaphor_permutations.csv')
    df.to_csv(outfile)


def do_permutation_test(probs_a, probs_b, n_samples=20000):
    # get the difference and significance of a - b
    mean_diff = np.mean(probs_a) - np.mean(probs_b)
    combined_probs = list(probs_a.copy()) + list(probs_b.copy())
    n_a = len(probs_a)
    rand_diffs = []
    # evaluate significance using the permutation test
    for i in tqdm(range(n_samples)):
        np.random.shuffle(combined_probs)
        rand_diffs.append(np.mean(combined_probs[:n_a]) - np.mean(combined_probs[n_a:]))
    # pvalue equals 1 - proportion of cases where observed is more extreme than random (opposite for opposite sign)
    if mean_diff > 0:
        mean_diff_pval = 1.0 - np.mean(np.array(rand_diffs < mean_diff))
    else:
        mean_diff_pval = 1.0 - np.mean(np.array(rand_diffs > mean_diff))

    return mean_diff, mean_diff_pval


if __name__ == '__main__':
    main()
