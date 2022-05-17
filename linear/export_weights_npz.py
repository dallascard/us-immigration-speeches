import os
import json
import joblib
from optparse import OptionParser

import numpy as np
import pandas as pd

from linear import file_handling as fh
from linear.train import LogisticRegression


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--model-file', type=str, default='model0.pkl',
                      help='model file name: default=%default')
    parser.add_option('--drop-feda', action="store_true", default=False,
                      help='Drop terms decorated with a feda feature: default=%default')

    (options, args) = parser.parse_args()

    model_file_name = options.model_file
    drop_feda = options.drop_feda

    model_dir = os.path.split(model_file_name)[0]
    outfile = model_file_name + '.weights.npz'


    config_file = os.path.join(model_dir, 'config.json')
    with open(config_file) as f:
        config = json.load(f)

    lower = config['model']['text_encoder']['lower']
    transform = config['model']['text_encoder']['transform']
    ngram_level = config['model']['text_encoder']['ngram_level']
    convert_digits = config['model']['text_encoder']['convert_digits']

    vocab_file = os.path.join(model_dir, 'vocab.json')
    label_file = os.path.join(model_dir, 'labels.json')
    model_file = os.path.join(model_dir, model_file_name)

    vocab = fh.read_json(vocab_file)
    labels = fh.read_json(label_file)
    model = joblib.load(model_file)

    coefs = model.coef_
    print("All weights:", coefs.shape)

    # drop zeros
    coef_abs_sums = np.sum(np.abs(coefs), axis=0)
    print("Total non-zero weights", np.sum(np.abs(coefs) > 0))
    valid_indices = [i for i, c in enumerate(coef_abs_sums) if c > 0]
    vocab = [vocab[i] for i in valid_indices]
    coefs = coefs[:, valid_indices]
    print("Non-zero weighted features", coefs.shape)

    if options.drop_feda:
        valid_indices = [i for i, v in enumerate(vocab) if '__' not in v]
        vocab = [vocab[i] for i in valid_indices]
        coefs = coefs[:, valid_indices]
        print("Generic terms:", coefs.shape)
        print("Non-zero generic weights", np.sum(np.abs(coefs) > 0))

    np.savez(outfile,
             weights=coefs,
             biases=model.intercept_,
             labels=labels,
             vocab=vocab,
             lower=lower,
             transform=transform,
             ngram_level=ngram_level,
             convert_digits=convert_digits)


if __name__ == '__main__':
    main()
