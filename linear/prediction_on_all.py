import os
import json
import joblib
from optparse import OptionParser

from linear.train import predict, LogisticRegression
from linear.docs import encode_documents_as_bow, load_data_directly
from linear.labels import encode_labels
from linear import file_handling as fh


# Do relevance/tone predictions on all segments from all congresses

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--model-dir', type=str, default='/u/scr/dcard/immigration/data/validation/relevance/exp/label/partition_v0.0_t0.0_s42/linear_f1_binarize_n2_l1/',
                      help='Override tokens field: default=%default')
    parser.add_option('--segments-dir', type=str, default='/u/scr/dcard/congress/segments/',
                      help='Segments dir: default=%default')
    parser.add_option('--uscr-segments-dir', type=str, default='/u/scr/dcard/congress/uscr_segments/',
                      help='USCR Segments dir: default=%default')
    parser.add_option('--first', type=int, default=43,
                      help='First congressional session: default=%default')
    parser.add_option('--last', type=int, default=116,
                      help='Last congressional session: default=%default')
    parser.add_option('--uscr-transition', type=int, default=112,
                      help='Congress at which to start using USCR: default=%default')
    parser.add_option('--tokens-field', type=str, default=None,
                      help='Override tokens field: default=%default')
    parser.add_option('--split-text', action="store_true", default=False,
                      help='Instead do simple white space splitting of text in tokens_field_name: default=%default')
    parser.add_option('--prefix', type=str, default=None,
                      help='Output prefix: default=%default')
    parser.add_option('--rel-file', type=str, default=None,
                      help='non_keyword_segment_probs.json (for predicting tone): default=%default')
    parser.add_option('--eval', action="store_true", default=False,
                      help='Evaluate predictions: default=%default')

    (options, args) = parser.parse_args()

    model_dir = options.model_dir
    segments_dir = options.segments_dir
    uscr_segments_dir = options.uscr_segments_dir
    first = options.first
    last = options.last
    uscr_transition = options.uscr_transition

    #split = options.split
    tokens_field = options.tokens_field
    split_text = options.split_text
    rel_file = options.rel_file
    eval = options.eval

    model_file = os.path.join(model_dir, 'model.nontest.pkl')
    print("Loading model")
    model = joblib.load(model_file)

    #model_dir = os.path.split(model_file)[0]
    config_file = os.path.join(model_dir, 'config.json')
    config = fh.read_json(config_file)
    vocab_file = os.path.join(model_dir, 'vocab.json')
    vocab = fh.read_json(vocab_file)
    label_file = os.path.join(model_dir, 'labels.json')
    label_vocab = fh.read_json(label_file)

    # combine text fields if desired
    dataset_reader = config["dataset_reader"]
    if tokens_field is None:
        tokens_field = dataset_reader['tokens_field_name']
    else:
        config['dataset_reader']['tokens_field_name'] = tokens_field
    if split_text:
        config['dataset_reader']['split_text'] = True
    print("Using tokens field:", tokens_field)
    print("Using label field:", dataset_reader['label_field_name'])

    if rel_file is not None:
        print("Reading relevant keyword segments", rel_file)
        with open(rel_file) as f:
            relevant_nonkeyword_segments = json.load(f)
        print("Read {:d} items".format(len(relevant_nonkeyword_segments)))

    for congress in range(first, last+1):
        print(congress)
        if congress < uscr_transition:
            infile = os.path.join(segments_dir, 'segments-' + str(congress).zfill(3) + '.jsonlist')        
        else:
            infile = os.path.join(uscr_segments_dir, 'segments-' + str(congress).zfill(3) + '.jsonlist')        

        basename = os.path.splitext(os.path.basename(infile))[0]

        output_prefix = options.prefix
        if output_prefix is None:
            output_prefix = basename

        n_examples = 1
        if rel_file is not None:
            print("Reading infile", infile)
            with open(infile) as f:
                temp = f.readlines()
            print("Read {:d} lines".format(len(temp)))

            lines = []
            for line in temp:
                line = json.loads(line)
                if line['id'] in relevant_nonkeyword_segments:
                    lines.append(line)
            print("Found {:d} relevant segments".format(len(lines)))    

            n_examples = len(lines)

            temp_file = infile[:-9] + '.temp.jsonlist'
            print("Saving relevant segments as", temp_file)
            with open(temp_file, 'w') as f:
                for line in lines:
                    f.write(json.dumps(line) + '\n')
            infile = temp_file

        if n_examples > 0:

            print("Loading data")
            train_docs, dev_docs, test_docs = load_data_directly(infile)
            docs = train_docs

            ids, line_indices, counts, _, instance_weights = encode_documents_as_bow(docs, vocab, config)

            print(counts.shape)

            if eval:
                labels, _, _ = encode_labels(docs, label_vocab, config)
            else:
                labels = None

            print("Predicting and saving to", model_dir)
            predict([model], counts, labels, instance_weights, ids, line_indices, label_vocab, model_dir, output_prefix)
            

if __name__ == '__main__':
    main()
