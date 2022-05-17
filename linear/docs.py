import re
import json
from collections import Counter

import numpy as np
from scipy import sparse

from linear import file_handling as fh
from linear.vocab import extract_vocab_params, convert_to_ngrams


def load_data(partition_file):

    partition = fh.read_json(partition_file)

    train_path = partition['train_file']
    dev_path = partition['dev_file']
    test_path = partition['test_file']

    train_indices = set(partition['train_indices'])
    if 'dev_indices' in partition:
        if partition['dev_indices'] is None:
            dev_indices = []
        else:
            dev_indices = partition['dev_indices']
    else:
        dev_indices = []
    if 'test_indices' in partition:
        if partition['test_indices'] is None:
            test_indices = []
        else:
            test_indices = set(partition['test_indices'])
    else:
        test_indices = []

    train_docs, dev_docs, test_docs = load_data_directly(train_path, train_indices, dev_path, dev_indices, test_path, test_indices)

    return train_docs, dev_docs, test_docs


def load_data_directly(train_path, train_indices=None, dev_path=None, dev_indices=None, test_path=None, test_indices=None):
    # Load data directly from a data file rather than a partition file (for prediction)

    train_docs = []
    dev_docs = []
    test_docs = []
    with open(train_path) as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            line['_i'] = 'tr_' + str(i)
            if train_indices is None or i in train_indices:
                train_docs.append(line)
            if dev_path == train_path and i in dev_indices:
                dev_docs.append(line)
            if test_path == train_path and i in test_indices:
                test_docs.append(line)

    if dev_path is not None and dev_path != train_path:
        dev_docs = load_subset(dev_path, dev_indices, 'dev')

    if test_path is not None and test_path != train_path:
        test_docs = load_subset(test_path, test_indices, 'test')

    return train_docs, dev_docs, test_docs        


def load_subset(path, indices=None, subset='na'):
    docs = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            line['_i'] = subset + '_' + str(i)
            if indices is None or i in indices:
                docs.append(line)
    return docs


def encode_documents_as_bow(documents, vocab, config, idf=None, truncate_feda=10):
    ngram_level, _, _, transform, lower, digits, exclude_nonalpha, require_alpha = extract_vocab_params(config)

    dataset_reader = config["dataset_reader"]
    tokens_field_name = dataset_reader['tokens_field_name']
    split_text = dataset_reader.get('split_text', False)
    weight_field_name = dataset_reader['weight_field_name']
    feda = dataset_reader['feda']

    vocab_size = len(vocab)
    vocab_index = dict(zip(vocab, range(vocab_size)))

    n_docs = len(documents)

    ids = []
    orig_indices = []

    n_features = vocab_size

    counts = sparse.lil_matrix((n_docs, n_features))
    weights = np.ones(n_docs)

    for i, doc in enumerate(documents):
        if 'id' in doc:
            ids.append(doc['id'])
        else:
            ids.append(doc['_i'])
        orig_indices.append(doc['_i'])
        text = doc[tokens_field_name]
        if split_text:
            text = [text.split()]
        token_counts = Counter()
        for sentence in text:
            try:
                assert(type(sentence) == list)
            except AssertionError as e:
                print("Input tokens should be a list of lists, not a list of strings!")
                raise e
            if lower:
                sentence = [token.lower() for token in sentence]
            if digits:
                sentence = [re.sub(r'\d', '#', token) for token in sentence]
            sentence = [token if re.match(r'[a-zA-Z0-9#$!?%"]+', token) is not None else '_' for token in sentence]
            token_counts.update(sentence)
            if feda is not None:
                # create duplicate features a la frustratingly easy domain adaptation
                feda_value = doc[feda]
                decorated = [token + '__' + str(feda_value)[:truncate_feda] for token in sentence if re.match(r'[a-zA-Z0-9#$!?%&"]+', token) is not None]
                token_counts.update(decorated)
            for n in range(2, ngram_level+1):
                ngrams = convert_to_ngrams(sentence, n, exclude_nonalpha, require_alpha)
                token_counts.update(ngrams)
                if feda is not None:
                    feda_value = doc[feda]
                    decorated = [ngram + '__' + str(feda_value)[:truncate_feda] for ngram in ngrams if re.match(r'[a-zA-Z0-9#$!?%&"]+', ngram) is not None]
                    token_counts.update(decorated)
        if transform == 'binarize':
            index_count_pairs = {vocab_index[term]: 1 for term, count in token_counts.items() if term in vocab_index}
        else:
            index_count_pairs = {vocab_index[term]: count for term, count in token_counts.items() if term in vocab_index}

        if len(index_count_pairs) > 0:
            indices, item_counts = zip(*index_count_pairs.items())
            counts[i, indices] = item_counts

        if weight_field_name is not None:
            if weight_field_name in doc:
                weights[i] = doc[weight_field_name]

    if transform == 'tfidf':
        if idf is None:
            print("Computing idf")
            idf = float(n_docs) / (np.array(counts.sum(0)).reshape((n_features, )))
        counts = counts.multiply(idf)

    counts = counts.tocsr()
    return ids, orig_indices, counts, idf, weights

