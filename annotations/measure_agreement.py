import json
from collections import Counter
from optparse import OptionParser

import numpy as np


def main():
    usage = "%prog infile.jsonlist"
    parser = OptionParser(usage=usage)
    parser.add_option('--response-field', type=str, default='response',
                      help='Name of column with ratings / responses: default=%default')
    parser.add_option('--worker-field', type=str, default='worker',
                      help='Name of column with worker (annotator) names / ids: default=%default')
    parser.add_option('--workers', type=str, default=None,
                      help='Optional comma-separated list of annotators to compare: default=%default')
    parser.add_option('--item-field', type=str, default='item',
                      help='Name of column with instance names / ids: default=%default')
    parser.add_option('--null-val', type=str, default='',
                      help='Response to interpret as null (in addition to None): default=%default')
    parser.add_option('--responses', type=str, default=None,
                      help='Optional comma-separated list of valid responses (mapped using edit distance): default=%default')
    parser.add_option('--outfile', type=str, default=None,
                      help='Optional output filename to save internal representations: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    infile = args[0]

    response_field = options.response_field
    item_field = options.item_field
    worker_field = options.worker_field
    valid_workers = options.workers
    null_val = str(options.null_val)
    valid_responses = options.responses
    outfile = options.outfile

    lines = []
    with open(infile) as f:
        for line in f:
            line = json.loads(line)
            lines.append(line)

    alpha = measure_agreement(lines, response_field, item_field, worker_field, valid_workers, null_val, valid_responses, outfile)
    print(alpha)


def measure_agreement(lines, response_field, item_field='item', worker_field='worker', valid_workers=None, null_val='', valid_responses=None, outfile=None, verbose=1):

    item_counter = Counter()
    worker_counter = Counter()
    response_counter = Counter()

    for line in lines:
        item_counter.update([str(line[item_field])])
        worker_counter.update([str(line[worker_field])])
        if response_field in line and line[response_field] is not None and str(line[response_field]) != null_val:
            response_counter.update([str(line[response_field])])
        else:
            response_counter.update([null_val])

    if verbose > 0:
        print("Read {:d} response lines".format(len(lines)))
        print("{:d} unique items".format(len(item_counter)))
        print("{:d} unique workers".format(len(worker_counter)))
        for w, c in worker_counter.most_common():
            print(w, c)
        print("{:d} unique responses".format(len(response_counter)))
        for r, c in response_counter.most_common():
            print(r, c)

    if valid_responses is not None:
        valid_responses = valid_responses.split(',')
        if verbose > 0:
            print("Using valid responses:")
            for v in valid_responses:
                print(v)

        if verbose > 0:
            print("Matching response to valid responses")
        response_map = {}
        for response in response_counter:
            if response != null_val:
                dists = []
                for possible_match in valid_responses:
                    dist = levenshtein_distance(response, possible_match)
                    dists.append(dist)
                closest_match = valid_responses[np.argmin(dists)]
                response_map[response] = closest_match
                if verbose > 0:
                    print(response, '->', closest_match)
    else:
        valid_responses = [r for r in response_counter.keys() if r != null_val]
        response_map = {r: r for r in response_counter.keys() if r != null_val}

    n_items = len(item_counter)
    n_workers = len(worker_counter)

    # map unique item, worker, and response values to integers
    item_dict = dict(zip(sorted(item_counter), range(n_items)))
    worker_dict = dict(zip(sorted(worker_counter), range(n_workers)))
    response_dict = dict(zip(sorted(valid_responses), range(1, len(valid_responses)+1)))
    response_map[null_val] = null_val
    response_dict[null_val] = 0

    if valid_workers is not None:
        valid_workers = valid_workers.split(',')
        if verbose > 0:
            print("Limiting analysis to workers:", ' '.join(valid_workers))
        worker_dict = dict(zip(sorted(valid_workers), range(len(valid_workers))))

    # Create item worker response matrix and initialize all entries to -1 (for non-respones)
    item_worker_response_matrix = np.zeros([n_items, n_workers], dtype=int)

    if outfile is not None:
        np.savez(outfile, data=item_worker_response_matrix, item=item_dict, workers=worker_dict, responses=response_dict)

    for line in lines:
        if response_field in line and line[response_field] is not None and line[response_field] != null_val:
            item_index = item_dict[str(line[item_field])]
            worker = str(line[worker_field])
            if worker in worker_dict:
                worker_index = worker_dict[worker]
                response = str(line[response_field])
                response_index = response_dict[response_map[response]]
                item_worker_response_matrix[item_index, worker_index] = response_index

    alpha = krippendorff_alpha_nominal(item_worker_response_matrix)
    return alpha


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def krippendorff_alpha_nominal(item_worker_response_matrix, verbose=1):
    # Calculate Krippendorff's alpha for nominal values, with 0 indicating non-response

    # Only use columns with more than 1 rating
    row_totals = np.sum(item_worker_response_matrix > 0, 1)
    row_sel = row_totals > 1
    n = np.sum(row_totals * row_sel)
    if verbose > 0:
        print("Found {:d} pairable items and {:d} pairable responses".format(int(row_sel.sum()), int(n)))
    item_worker_response_matrix = item_worker_response_matrix[row_sel, :]

    n_items, n_workers = np.shape(item_worker_response_matrix)

    item_disagreements = np.array([calc_disagreements(np.bincount(item_worker_response_matrix[i, :])) for i in range(n_items)])
    item_counts = np.array([np.sum(item_worker_response_matrix[i, :] != 0) for i in range(n_items)], dtype=float)

    D_o = np.sum(item_disagreements / (item_counts-1)) / float(n)

    counts = np.bincount(item_worker_response_matrix.reshape(n_items * n_workers))
    D_e = calc_disagreements(counts) / (n*(n-1))

    alpha = 1 - D_o / D_e

    return alpha


def calc_disagreements(bin_count):
    XXt = np.outer(bin_count, bin_count)
    XXt -= np.diag(np.diag(XXt))
    n_disagreements = np.sum(XXt[1:,1:])/2.0
    return n_disagreements


if __name__ == '__main__':
    main()
