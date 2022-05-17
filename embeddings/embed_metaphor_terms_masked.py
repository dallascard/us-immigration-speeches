import os
import re
import json
from collections import Counter
from optparse import OptionParser

import torch
import numpy as np
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer

from analysis.metaphor_terms import get_metaphor_terms

# Export token occurrences (masked) to vector occurrences using Roberta


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--indir', type=str, default='data/speeches/Congress/metaphors/',
                      help='Directory with output of analysis.identify_metaphor_mentions.py: default=%default')
    parser.add_option('--model-type', type=str, default='bert',
                      help='Model type: default=%default')
    parser.add_option('--tokenizer', type=str, default=None,
                      help='Tokenizer name or path (defaults to model): default=%default')
    parser.add_option('--model', type=str, default='bert-base-uncased',
                      help='Model name or path: default=%default')
    parser.add_option('--device', type=int, default=0,
                      help='GPU to use: default=%default')
    parser.add_option('--outdir', type=str, default='data/speeches/Congress/metaphors/contextual-embeddings/',
                      help='Model name or path: default=%default')

    (options, args) = parser.parse_args()

    indir = options.indir
    tokenizer = options.tokenizer
    model_type = options.model_type
    model_name_or_path = options.model
    device = options.device
    outdir = options.outdir

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    metaphor_terms = get_metaphor_terms()
    metaphor_term_sets = {}
    for metaphor, terms in metaphor_terms.items():
        metaphor_term_sets[metaphor] = set(terms)

    print("Loading model")
    if model_type == 'bert':
        model_class = BertModel
        tokenizer_class = BertTokenizer
    elif model_type == 'roberta':
        model_class = RobertaModel
        tokenizer_class = RobertaTokenizer
    else:
        raise ValueError("Model type not recognized")

    # Load pretrained model/tokenizer
    if tokenizer is None:
        tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    else:
        tokenizer = tokenizer_class.from_pretrained(tokenizer)

    model = model_class.from_pretrained(model_name_or_path)

    # move the model to the GPU
    torch.cuda.set_device(device)
    device = torch.device("cuda", device)
    model.to(device)

    print("Reading input data")
    for metaphor in sorted(metaphor_term_sets):

        line_indices = []
        speech_ids = []
        sent_indices = []
        last_layer_vectors = []
        term_counter = Counter()

        masked_terms = []
        contexts = []
        tokenized_contexts = []
        word_piece_indices = []

        target_terms = metaphor_term_sets[metaphor]
        infile = os.path.join(indir, 'metaphor_mention_sents_parsed_{:s}.jsonlist'.format(metaphor))
        print(metaphor, infile)
        with open(infile) as f:
            lines = f.readlines()

        print("Export embeddings")
        for line_i, line in enumerate(tqdm(lines)):
            line = json.loads(line)
            speech_id = line['id']
            sent_index = line['sent_index']
            orig_tokens = line['tokens']
            text = ' '.join([t.lower() for t in orig_tokens]).strip()
            # get the text with search terms already converted to single tokens
            #simplified_text = line['simplified']

            # encode the simplified text (with replacements made)
            input_ids = tokenizer.encode(text, max_length=512, truncation=True)

            # get the tokenized representation from the contextual emebdding model
            tokens = [tokenizer.ids_to_tokens[i] for i in input_ids]

            # mask out the target terms, and store indices (and identities) of target terms
            target_indices = []
            masked_tokens = []
            for t_i, token in enumerate(tokens[:-1]):
                if token in target_terms and not tokens[t_i+1].startswith('##'):
                    masked_tokens.append(tokenizer.mask_token)
                    target_indices.append(t_i)
                else:
                    masked_tokens.append(token)
            masked_tokens.append(tokens[-1])

            masked_text = ' '.join(masked_tokens)
            masked_text = re.sub(' ##', '', masked_text)

            if len(target_indices) > 0:

                input_ids = tokenizer.encode(masked_tokens, add_special_tokens=False)
                input_ids = torch.tensor([input_ids])
                input_ids_on_device = input_ids.to(device)

                # process the text through the model
                with torch.no_grad():
                    try:
                        last_layer = model(input_ids_on_device)[0]

                        last_layer_np = last_layer.detach().cpu().numpy()

                        for i in target_indices:
                            line_indices.append(line_i)
                            speech_ids.append(speech_id)
                            sent_indices.append(sent_index)
                            vector = last_layer_np[0, i, :].copy()
                            last_layer_vectors.append(vector)

                            masked_terms.append(tokens[i])
                            contexts.append(masked_text)
                            tokenized_contexts.append(masked_tokens)
                            word_piece_indices.append(i)

                    except Exception as e:
                        raise e

        print("Concatenating vectors")
        last_layer_vectors = np.vstack(last_layer_vectors)

        print(last_layer_vectors.shape, len(masked_terms))

        print("Saving data")
        np.savez_compressed(os.path.join(outdir, 'metaphor_vectors_masked_{:s}.npz'.format(metaphor)), vectors=last_layer_vectors)

        with open(os.path.join(outdir, 'masked_terms_{:s}.json'.format(metaphor)), 'w') as f:
            json.dump(masked_terms, f, indent=2)

        with open(os.path.join(outdir, 'line_indices_{:s}.json'.format(metaphor)), 'w') as f:
            json.dump(line_indices, f, indent=2)

        with open(os.path.join(outdir, 'speech_ids_{:s}.json'.format(metaphor)), 'w') as f:
            json.dump(speech_ids, f, indent=2)

        with open(os.path.join(outdir, 'sent_indices_{:s}.json'.format(metaphor)), 'w') as f:
            json.dump(sent_indices, f, indent=2)

        with open(os.path.join(outdir, 'word_piece_indices_{:s}.json'.format(metaphor)), 'w') as f:
            json.dump(word_piece_indices, f, indent=2)

        with open(os.path.join(outdir, 'masked_contexts_{:s}.json'.format(metaphor)), 'w') as f:
            json.dump(contexts, f, indent=2)

        with open(os.path.join(outdir, 'masked_contexts_tokenized_{:s}.json'.format(metaphor)), 'w') as f:
            json.dump(tokenized_contexts, f, indent=2)

        print("Term counts:")
        term_counter.update(masked_terms)
        for term, count in term_counter.most_common():
            print(term, count)


if __name__ == '__main__':
    main()
