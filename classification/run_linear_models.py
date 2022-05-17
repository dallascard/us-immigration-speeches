import os
from optparse import OptionParser
from subprocess import call


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    #parser.add_option('--issue', type=str, default='immigration',
    #                  help='Issue: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    n_folds = 10
    datasets = ['basic', 'basic-tsw', 'label-weights', 'label-weights-tsw']

    for dataset in datasets:
        for fold in range(n_folds):
            train_file = os.path.join('data', 'congress', 'relevance', dataset, 'folds', str(fold), 'train.jsonlist')
            dev_file = os.path.join('data', 'congress', 'relevance', dataset, 'folds', str(fold), 'dev.jsonlist')
            test_file = os.path.join('data', 'congress', 'relevance', dataset, 'folds', str(fold), 'test.jsonlist')

            # make the partition
            cmd = ['python', '-m', 'partition.create', train_file, 'label', '--test', test_file, '--dev', dev_file]
            print(' '.join(cmd))
            call(cmd)

            # run the model
            partition_file = os.path.join('data', 'congress', 'relevance', dataset, 'folds', str(fold), 'exp', 'label', 'partition_v0.0_t0.0_s42', 'partition.json')
            cmd = ['python', '-m', 'models.lr', partition_file, '--weight-field', 'weight', '--run', '--metric', 'f1', '--penalty', 'l2']
            call(cmd)


if __name__ == '__main__':
    main()
