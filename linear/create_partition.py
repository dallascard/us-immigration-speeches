import os
from optparse import OptionParser

import numpy as np

from linear import file_handling as fh

# Utility script to tell guac which data to use for train/dev/test and/or create random splits
# Option 1: provide a train dev and test file, already split
# Option 2: provide a single file, and a proportion to use for dev and test (rest for train)
# Option 3: provide a single file, and a number of test and dev folds (for CV)


def main():
    usage = "%prog data.jsonlist task"
    parser = OptionParser(usage=usage)
    parser.add_option('--name', type=str, default='partition',
                      help='Base name: default=%default')
    parser.add_option('--dev', type=str, default=None,
                      help='Path to separate dev file.jsonlist: default=%default')
    parser.add_option('--test', type=str, default=None,
                      help='Path to separate test file.jsonlist: default=%default')
    parser.add_option('--unlabeled', type=str, default=None,
                      help='Path to unlabeled file.jsonlist: default=%default')
    parser.add_option('--test-prop', type=float, default=0.,
                      help='Number of test folds: default=%default')
    parser.add_option('--dev-prop', type=float, default=0.,
                      help='Number of dev folds: default=%default')
    parser.add_option('--dev-subset', type=str, default=None,
                      help='Use a subset of train as dev (e.g. issue=immigration): default=%default')
    parser.add_option('--test-folds', type=int, default=1,
                      help='Number of test folds [NOT YET IMPLEMENTED]: default=%default')
    parser.add_option('--dev-folds', type=int, default=1,
                      help='Number of dev folds [NOT YET IMPLEMENTED]: default=%default')
    parser.add_option('--stratify', action="store_true", default=False,
                      help='Stratify splits by label [NOT YET IMPLEMENTED]: default=%default')
    parser.add_option('--seed', type=int, default=42,
                      help='Random seed: default=%default')

    (options, args) = parser.parse_args()

    data_file = args[0]
    task = args[1]

    name = options.name
    dev_file = options.dev
    test_file = options.test
    unlabeled_file = options.unlabeled
    test_prop = options.test_prop
    dev_prop = options.dev_prop
    dev_subset = options.dev_subset
    test_folds = options.test_folds
    dev_folds = options.dev_folds
    stratify = options.stratify
    seed = options.seed

    create_partition(data_file, task, name, dev_file, test_file, unlabeled_file, test_prop, dev_prop, dev_subset, test_folds, dev_folds, stratify, seed)


def create_partition(data_file, task, name='partition', dev_file=None, test_file=None, unlabeled_file=None, test_prop=0., dev_prop=0., dev_subset=None, test_folds=1, dev_folds=1, stratify=False, seed=42):

    np.random.seed(seed)

    output = {}

    if dev_file is not None and test_file is not None:
        print("Using pre-determined train/dev/test split")
        print("Reading train data from", data_file)
        train = fh.read_jsonlist(data_file)
        train_indices = [i for i, line in enumerate(train) if task in line]
        print("Reading dev data from", dev_file)
        dev = fh.read_jsonlist(dev_file)
        dev_indices = [i for i, line in enumerate(dev) if task in line]
        print("Reading test data from", test_file)
        test = fh.read_jsonlist(test_file)
        test_indices = [i for i, line in enumerate(test) if task in line]
        print("Found {:d}/{:d}/{:d} train/dev/test instances".format(len(train), len(dev), len(test)))
        print("Found {:d}/{:d}/{:d} train/dev/test instances with labels".format(len(train_indices), len(dev_indices), len(test_indices)))

        output['train_file'] = data_file
        output['dev_file'] = dev_file
        output['test_file'] = test_file
        output['unlabeled_file'] = unlabeled_file
        output['dev_folds'] = 1
        output['stratified'] = False
        output['train_indices'] = train_indices
        output['dev_indices'] = dev_indices
        output['test_indices'] = test_indices

    elif dev_file is None and test_file is not None:
        print("Using pre-determined train/test split")
        print("Reading train data from", data_file)
        train = fh.read_jsonlist(data_file)
        train_indices = [i for i, line in enumerate(train) if task in line]
        print("Reading test data from", test_file)
        test = fh.read_jsonlist(test_file)
        test_indices = [i for i, line in enumerate(test) if task in line]
        print("Found {:d}/{:d} train/test instances".format(len(train), len(test)))
        print("Found {:d}/{:d} train/test instances with labels".format(len(train_indices), len(test_indices)))

        if dev_prop > 0:
            n_dev = int(np.round(len(train_indices) * dev_prop))
            print("Selecting {:d} dev instances".format(n_dev))
            dev_indices = np.random.choice(train_indices, size=n_dev, replace=False)
            dev_indices = [int(i) for i in dev_indices]
            dev_set = set(dev_indices)
            train_indices = [i for i in train_indices if i not in dev_set]
        elif dev_subset is not None:
            field, value = dev_subset.split('=')
            print("Selecting dev instances with {:s} == {:s}".format(field, value))
            dev_indices = [i for i, line in enumerate(train) if line[field] == value]
            dev_set = set(dev_indices)
            train_indices = [i for i in train_indices if i not in dev_set]
            print("Train/dev: {:d}/{:d}".format(len(train_indices), len(dev_indices)))
        else:
            dev_indices = None

        output['train_file'] = data_file
        output['dev_file'] = data_file
        output['test_file'] = test_file
        output['unlabeled_file'] = unlabeled_file
        output['dev_folds'] = 1
        output['stratified'] = False
        output['train_indices'] = train_indices
        output['dev_indices'] = dev_indices
        output['test_indices'] = test_indices

    elif dev_file is None and test_file is None:
        print("Reading data from", data_file)
        data = fh.read_jsonlist(data_file)
        print("Found {:d} instances".format(len(data)))
        data_indices = [i for i, line in enumerate(data) if task in line]
        print("Found {:d} instances with labels".format(len(data_indices)))
        if test_prop > 0:
            n_test = int(np.round(len(data_indices) * test_prop))
            print("Selecting {:d} test instances".format(n_test))
            test_indices = np.random.choice(data_indices, size=n_test, replace=False)
            # convert to basic ints for serialization
            test_indices = [int(i) for i in test_indices]
            test_set = set(test_indices)
            nontest_indices = [i for i in data_indices if i not in test_set]
        else:
            test_indices = []
            nontest_indices = data_indices
        if dev_prop > 0:
            n_dev = int(np.round(len(nontest_indices) * dev_prop))
            print("Selecting {:d} dev instances".format(n_dev))
            dev_indices = np.random.choice(nontest_indices, size=n_dev, replace=False)
            dev_indices = [int(i) for i in dev_indices]
            dev_set = set(dev_indices)
            train_indices = [i for i in nontest_indices if i not in dev_set]
        else:
            dev_indices = []
            train_indices = nontest_indices

        output['train_file'] = data_file
        output['dev_file'] = data_file
        output['test_file'] = data_file
        output['unlabeled_file'] = unlabeled_file
        output['dev_folds'] = 1
        output['stratified'] = False
        output['train_indices'] = train_indices
        output['dev_indices'] = dev_indices
        output['test_indices'] = test_indices

    output['task'] = task
    output['seed'] = seed

    partition_name = name + '_'
    if dev_subset is None:
        partition_name += 'v' + str(dev_prop) + '_t' + str(test_prop) + '_s' + str(seed)
    else:
        partition_name += dev_subset + '_t' + str(test_prop) + '_s' + str(seed)
    output_dir = os.path.join(os.path.split(data_file)[0], 'exp', task, partition_name)
    fh.makedirs(output_dir)
    output_file = os.path.join(output_dir, 'partition.json')

    fh.write_to_json(output, output_file, indent=None, sort_keys=False)


if __name__ == '__main__':
    main()
