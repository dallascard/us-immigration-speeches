import os
import glob
import shutil
import json
from subprocess import call
from optparse import OptionParser
from collections import defaultdict


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--model_type', type=str, default='roberta',
                      help='Model type: [bert|roberta|...] default=%default')
    parser.add_option('--model_name_or_path', type=str, default='roberta-base',
                      help='Path to model: [bert-base-uncased|roberta-base|...] default=%default')
    parser.add_option('--tokenizer_name', type=str, default='',
                      help='Tokenizer name [''|bert-base-uncased|etc] default=%default')
    parser.add_option('--lr', type=float, default=2e-5,
                      help='Learning rate: default=%default')
    parser.add_option('--max_seq_length', type=int, default=512,
                      help='max seq length: default=%default')
    parser.add_option('--per_gpu', type=int, default=4,
                      help='Examples per GPU: default=%default')
    parser.add_option('--n_epochs', type=int, default=7,
                      help='Number of epochs to run: default=%default')
    parser.add_option('--seed', type=int, default=42,
                      help='Random seed: default=%default')
    parser.add_option('--basedir', type=str, default='data/congress/tone/label-weights/all/',
                      help='Output directory from split_data.py: default=%default')
    parser.add_option('--train-file', type=str, default='all.jsonlist',
                      help='Name of train file: default=%default')
    parser.add_option('--output-prefix', type=str, default='all',
                      help='Output directory prefix: default=%default')
    #parser.add_option('--exp-dir', type=str, default='results',
    #                  help='Experiment output directory: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    seed = options.seed
    model_type = options.model_type
    model_name_or_path = options.model_name_or_path
    tokenizer_name = options.tokenizer_name
    lr = options.lr
    max_seq_length = options.max_seq_length
    per_gpu = options.per_gpu
    n_epochs = options.n_epochs
    train_file = options.train_file
    output_prefix = options.output_prefix

    run(basedir, seed, model_type, model_name_or_path, tokenizer_name, lr, max_seq_length, per_gpu, n_epochs, train_file=train_file, output_prefix=output_prefix)


def run(basedir, seed=42, model_type='bert', model_name_or_path='bert-base-uncased', tokenizer_name='', lr=2e-5, max_seq_length=512, per_gpu=4, n_epochs=7, train_file='all.jsonlist', output_prefix='all'):

    name = os.path.basename(basedir)

    if output_prefix is None:
        part1, part2 = os.path.split(model_name_or_path)
        if len(part2) == 0:
            part1, part2 = os.path.split(part1)
        output_prefix = part2
        assert len(part2) > 0
    output_prefix += '_s' + str(seed)
    output_prefix += '_lr' + str(lr)
    output_prefix += '_msl' + str(max_seq_length)
    print(output_prefix)

    config = {'basedir': basedir,
              'seed': int(seed),
              'model_type': model_type,
              'model_name_or_path': model_name_or_path,
              'lr': float(lr),
              'max_seq_length': int(max_seq_length),
              'train_file': train_file
              }

    outdir = os.path.join(basedir,  'bert', output_prefix)

    cmd = ['python', '-m', 'hf.run',
           '--model_type', model_type,
           '--model_name_or_path', model_name_or_path,
           '--tokenizer_name', tokenizer_name,
           '--name', name,
           '--do_train',
           '--train', train_file,
           '--data_dir', os.path.join(basedir),
           '--max_seq_length', str(max_seq_length),
           '--per_gpu_eval_batch_size=' + str(per_gpu),
           '--per_gpu_train_batch_size=' + str(per_gpu),
           '--learning_rate', str(lr),
           '--num_train_epochs', str(n_epochs),
           '--output_dir', outdir,
           '--overwrite_cache',
           '--overwrite_output_dir',
           '--weight_field', 'weight',
           '--metrics', 'accuracy,per_class_f1',
           '--seed', str(seed)
           ]
    if model_type == 'bert':
        cmd.append('--do_lower_case')

    print(' '.join(cmd))
    call(cmd)
    with open(os.path.join(outdir, 'train_cmd.txt'), 'w') as f:
        f.write(' '.join(cmd))

    with open(os.path.join(outdir, 'my_config.json'), 'w') as f:
        json.dump(config, f, indent=2, sort_keys=False)


if __name__ == '__main__':
    main()
