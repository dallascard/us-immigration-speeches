import os
import joblib
from optparse import OptionParser

import numpy as np
import pandas as pd

from linear import file_handling as fh
from linear.train import LogisticRegression


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--model-file', type=str, default='model.nontest.pkl',
                      help='model file name: default=%default')
    parser.add_option('--sep-columns', action="store_true", default=False,
                      help='Use a separate vocabulary file for each column: default=%default')

    (options, args) = parser.parse_args()

    model_file_name = options.model_file

    model_dir = os.path.split(model_file_name)[0]
    outfile = model_file_name + '.weights.tsv'

    vocab_file = os.path.join(model_dir, 'vocab.json')
    label_file = os.path.join(model_dir, 'labels.json')
    model_file = os.path.join(model_dir, model_file_name)

    vocab = fh.read_json(vocab_file)
    labels = fh.read_json(label_file)
    model = joblib.load(model_file)

    n_labels = len(labels)
    print("{:d} labels".format(n_labels))

    coefs = model.coef_
    biases = model.intercept_
    print("All weights:", coefs.shape)

    # drop zeros
    coef_abs_sums = np.sum(np.abs(coefs), axis=0)
    print("Total non-zero weights", np.sum(np.abs(coefs) > 0))
    valid_indices = [i for i, c in enumerate(coef_abs_sums) if c > 0]
    vocab = [vocab[i] for i in valid_indices]
    coefs = coefs[:, valid_indices]
    print("Non-zero weighted features", coefs.shape)

    if n_labels > 2:
        if options.sep_columns:
            df = pd.DataFrame(index=np.arange(len(vocab)+1))
            for label_i, label in enumerate(labels):
                order = np.argsort(coefs[label_i, :])[::-1]
                df[label + '_terms'] = ['__BIAS__'] + [vocab[i] for i in order]
                df[label] = [biases[label_i]] + [coefs[label_i, i] for i in order]
        else:
            df = pd.DataFrame(index=['__BIAS__'] + vocab)
            for label_i, label in enumerate(labels):
                df[label] = [biases[label_i]] + list(coefs[label_i, :])
    else:
        df = pd.DataFrame(index=['__BIAS__'] + vocab)
        df[labels[1]] = [biases[0]] + list(coefs[0, :])

    df.to_csv(outfile, sep='\t')


if __name__ == '__main__':
    main()
