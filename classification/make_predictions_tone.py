import os
import glob
import shutil
import json
from subprocess import call
from optparse import OptionParser
from collections import defaultdict


## Redundant with make_predictions.py, I think!


def main():
    usage = "%prog infile.jsonlist outfile.tsv"
    parser = OptionParser(usage=usage)
    #parser.add_option('--guac-dir', type=str, default='../guac/',
    #                  help='guac directory: default=%default')
    parser.add_option('--model_type', type=str, default='roberta',
                      help='Model type: [bert|roberta|...] default=%default')
    parser.add_option('--model_name_or_path', type=str, default='roberta-base',
                      help='Path to model: [bert-base-uncased|roberta-base|...] default=%default')
    parser.add_option('--tokenizer_name', type=str, default='',
                      help='Tokenizer name [''|bert-base-uncased|etc] default=%default')
    parser.add_option('--max_seq_length', type=int, default=512,
                      help='max seq length: default=%default')
    parser.add_option('--per_gpu', type=int, default=4,
                      help='Examples per GPU: default=%default')
    parser.add_option('--train-file', type=str, default='/u/scr/dcard/projects/guac/data/congress/relevance/basic/all/all.jsonlist',
                      help='File used to train model (to get labels): default=%default')
    #parser.add_option('--exp-dir', type=str, default='results',
    #                  help='Experiment output directory: default=%default')

    (options, args) = parser.parse_args()

    infile = args[0]
    outfile = args[1]

    model_type = options.model_type
    model_name_or_path = options.model_name_or_path
    tokenizer_name = options.tokenizer_name
    max_seq_length = options.max_seq_length
    per_gpu = options.per_gpu
    train_file = options.train_file

    make_predictions(infile, outfile, train_file, model_type, model_name_or_path, tokenizer_name, max_seq_length, per_gpu)


def make_predictions(infile, outfile, train_file, model_type='roberta', model_name_or_path='roberta-base', tokenizer_name='', max_seq_length=512, per_gpu=4, overwrite_output_dir=True):

    indir, _ = os.path.split(infile)
    outdir, outfile = os.path.split(outfile)
    train_dir, train_file = os.path.split(train_file)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    cmd = ['python', '-m', 'hf.run',
           '--name', 'best_test',
           '--model_type', model_type,
           '--model_name_or_path',  model_name_or_path,
           '--tokenizer', tokenizer_name,
           '--data_dir', train_dir,
           '--train', train_file,
           '--predict', infile,
           '--output_dir',  outdir,
           '--pred_file_name', outfile,
           '--max_seq_length', str(max_seq_length),
           '--per_gpu_eval_batch_size=' + str(per_gpu),
           '--per_gpu_train_batch_size=' + str(per_gpu),
           '--overwrite_cache',
           ]
    if model_type == 'bert':
        cmd.append('--do_lower_case')

    if overwrite_output_dir:
        cmd.append('--overwrite_output_dir')

    print(' '.join(cmd))
    call(cmd)


if __name__ == '__main__':
    main()
