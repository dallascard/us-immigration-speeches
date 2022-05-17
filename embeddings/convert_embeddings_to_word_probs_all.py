import os
import re
import json
from glob import glob
from collections import Counter, defaultdict
from optparse import OptionParser

import torch
import numpy as np
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer, BertForMaskedLM, RobertaForMaskedLM
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from scipy.special import softmax


# Export token occurrences (masked) to vector occurrences using Roberta


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--infile', type=str, default='data/speeches/Congress/contextual-embeddings/bert-base-uncased/immigrant_vectors_masked.npz',
                      help='Infile: default=%default')
    parser.add_option('--model-type', type=str, default='bert',
                      help='Model type: default=%default')
    parser.add_option('--tokenizer', type=str, default=None,
                      help='Tokenizer name or path (defaults to model): default=%default')
    parser.add_option('--model', type=str, default='bert-base-uncased',
                      help='Model name or path: default=%default')
    #parser.add_option('--batch-size', type=int, default=100,
    #                  help='Batch size: default=%default')
    #parser.add_option('--min-count', type=int, default=10,
    #                  help='Min count (over all years): default=%default')
    parser.add_option('--device', type=int, default=0,
                      help='GPU to use: default=%default')
    parser.add_option('--batch-size', type=int, default=256,
                      help='Batch size: default=%default')
    parser.add_option('--outfile', type=str, default='data/speeches/Congress/contextual-embeddings/bert-base-uncased/immigrant_vectors_maked_all_probs.npz',
                      help='Model name or path: default=%default')
    #parser.add_option('--target-terms', type=str, default='flood,floods,river,rivers,stream,steams,tide,tides,water,waters',
    #                  help='Model type: default=%default')

    (options, args) = parser.parse_args()

    infile = options.infile
    tokenizer = options.tokenizer
    model_type = options.model_type
    model_name_or_path = options.model
    device = options.device
    batch_size = options.batch_size
    outfile = options.outfile

    output_vectors = []

    print("Loading model")
    if model_type == 'bert':
        model_class = BertModel
        tokenizer_class = BertTokenizer
        lm_model_class = BertForMaskedLM
        lm_head_class = BertOnlyMLMHead
    elif model_type == 'roberta':
        model_class = RobertaModel
        tokenizer_class = RobertaTokenizer
        lm_model_class = RobertaForMaskedLM
        raise NotImplementedError("Need to get the lm_head_class for Roberta")
        #lm_head_class = transformers.modeling_bert.BertOnlyMLMHead
    else:
        raise ValueError("Model type not recognized")

    # Load pretrained model/tokenizer
    if tokenizer is None:
        tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    else:
        tokenizer = tokenizer_class.from_pretrained(tokenizer)

    #model = model_class.from_pretrained(model_name_or_path)
    lm = lm_model_class.from_pretrained(model_name_or_path)

    # move the model to the GPU
    torch.cuda.set_device(device)
    device = torch.device("cuda", device)
    lm.to(device)

    # get the MLM head
    mlm = None
    for m in lm.modules():
        if type(m) == lm_head_class:
            mlm = m

    embeddings = np.load(infile)['vectors']
    n_words, dim = embeddings.shape

    n_batches = n_words // batch_size + 1
    print(n_batches)

    for b in tqdm(range(n_batches)):
        batch = embeddings[b * batch_size: (b + 1) * batch_size, :]
        batch = torch.tensor(batch).to(device)
        preds = mlm(batch)
        probs = torch.softmax(preds, dim=1)
        probs_np = probs.detach().cpu().numpy().copy()
        output_vectors.append(probs_np)

    np.savez(outfile, probs=np.vstack(output_vectors))


if __name__ == '__main__':
    main()
