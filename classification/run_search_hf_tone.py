import os
from optparse import OptionParser

import numpy as np

from congress.run_folds_hf_tone import run_folds


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--folds', type=int, default=5,
                      help='Number of test folds: default=%default')
    parser.add_option('--max_seq_length', type=int, default=512,
                      help='max seq length: default=%default')
    #parser.add_option('--do_text_b', action="store_true", default=False,
    #                  help='Use second sequence: default=%default')
    parser.add_option('--seed', type=int, default=84,
                      help='Random seed: default=%default')
    #parser.add_option('--save-steps', type=int, default=94,
    #                  help='Steps between each save: default=%default')
    parser.add_option('--model_type', type=str, default='roberta',
                      help='Model type: [bert|roberta|...] default=%default')
    parser.add_option('--model_name_or_path', type=str, default='roberta-base',
                      help='Path to model: [bert-base-uncased|roberta-base|...] default=%default')
    parser.add_option('--tokenizer_name', type=str, default='',
                      help='Tokenizer name [''|bert-base-uncased|etc] default=%default')
    parser.add_option('--basedir', type=str, default='data/speeches/Congress/tone/',
                      help='Base directory from split_data.py: default=%default')
    #parser.add_option('--transformers-dir', type=str, default='/u/scr/dcard/tools/transformers/',
    #                  help='Transformers dir: default=%default')
    parser.add_option('--n-seeds', type=int, default=8,
                      help='Number of random seeds to try: default=%default')
    parser.add_option('--first-seed', type=int, default=0,
                      help='Seed to start at: default=%default')
    parser.add_option('--last-seed', type=int, default=7,
                      help='Last seed to run: default=%default')
    parser.add_option('--per_gpu', type=int, default=4,
                      help='Examples per GPU: default=%default')
    parser.add_option('--n_epochs', type=int, default=7,
                      help='Number of epochs to run: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir

    start_seed = options.seed
    folds = options.folds
    max_seq_length = options.max_seq_length
    model_type = options.model_type
    model_name_or_path = options.model_name_or_path
    tokenizer_name = options.tokenizer_name
    n_seeds = options.n_seeds
    first_seed = options.first_seed
    last_seed = options.last_seed
    per_gpu = options.per_gpu
    n_epochs = options.n_epochs

    np.random.seed(start_seed)
    seeds = np.random.randint(low=0, high=2**32 - 1, size=n_seeds)

    for seed in seeds[first_seed:last_seed+1]:
        for split in ['basic', 'label-weight']:
            lr = 2e-5
            run_folds(basedir=os.path.join(basedir, split),
                      seed=seed,
                      model_type=model_type,
                      model_name_or_path=model_name_or_path,
                      tokenizer_name=tokenizer_name,
                      folds=folds,
                      start_fold=0,
                      lr=lr,
                      max_seq_length=max_seq_length,
                      per_gpu=per_gpu,
                      n_epochs=n_epochs)


if __name__ == '__main__':
    main()
