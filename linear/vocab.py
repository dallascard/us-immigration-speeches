import re
from collections import Counter

import numpy as np

import linear.file_handling as fh


def build_vocab(documents, config, truncate_feda=10):
    """
    Create an n-gram vocabulary from the tokenized sentences for each document
    """
    ngram_level, min_doc_freq, max_doc_prop, _, lower, digits, exclude_nonalpha, require_alpha = extract_vocab_params(config)

    dataset_reader = config["dataset_reader"]
    feda = dataset_reader['feda']
    tokens_field_name = dataset_reader['tokens_field_name']

    stopwords_file = config['model']['text_encoder']['stopwords_file']
    if stopwords_file is not None:
        print("Reading stopwords from", stopwords_file)
        stopwords = fh.read_text_to_list(stopwords_file)
        stopwords = set([word.strip() for word in stopwords])
    else:
        stopwords = set()

    unigram_counter = Counter()
    bigram_counter = Counter()
    doc_counts = Counter()
    n_docs = len(documents)
    for doc in documents:
        # get the set of words in this document
        text = doc[tokens_field_name]
        for sentence in text:
            # replace underscores with dashes
            sentence = [re.sub('_', '-', token) for token in sentence]
            # replace stopwords with underscores
            sentence = ['_' if token.lower() in stopwords else token for token in sentence]
            if lower:
                sentence = [token.lower() for token in sentence]
            if digits:
                sentence = [re.sub(r'\d', '#', token) for token in sentence]
            # remove most lone puncatuation except $, #, !, ?, %, and "
            sentence = [token if re.match(r'[a-zA-Z0-9#$!?%"]+', token) is not None else '_' for token in sentence]
            if exclude_nonalpha:
                sentence = [token if token.isalpha() else '_' for token in sentence]
            elif require_alpha:
                sentence = [token if re.match(r'.*[a-zA-Z0-9]+', token) is not None else '_' for token in sentence]
            unigram_counter.update(sentence)
            doc_counts.update(set(sentence))
            if feda is not None:
                # create duplicate features a la frustratingly easy domain adaptation
                feda_value = doc[feda]
                decorated = [token + '__' + str(feda_value)[:truncate_feda] for token in sentence if re.match(r'[a-zA-Z0-9#$!?%"]+', token) is not None]
                unigram_counter.update(decorated)
                doc_counts.update(set(decorated))
            for n in range(2, ngram_level+1):
                ngrams = convert_to_ngrams(sentence, n, exclude_nonalpha, require_alpha)
                doc_counts.update(set(ngrams))
                if n == 2:
                    bigram_counter.update(ngrams)
                if feda is not None:
                    # create duplicate features a la frustratingly easy domain adaptation
                    feda_value = doc[feda]
                    decorated = [ngram + '__' + str(feda_value)[:truncate_feda] for ngram in ngrams if re.match(r'[a-zA-Z0-9#$!?%&"]+', ngram) is not None]
                    doc_counts.update(set(decorated))
                    if n == 2:
                        bigram_counter.update(decorated)

    print("# unigrams:", len(unigram_counter))
    if ngram_level > 1:
        print("# bigrams:", len(bigram_counter))

    print('all features:', len(doc_counts))

    print("Filtering on frequency")
    doc_counts = {k: v for k, v in doc_counts.items() if v/float(n_docs) <= max_doc_prop}
    print(len(doc_counts))
    vocab = [word for word, count in doc_counts.items() if count >= min_doc_freq]
    if '_' in vocab:
        vocab.remove('_')
    print("Vocab size = {:d}".format(len(vocab)))

    vocab.sort()
    print("Final vocab size = {:d}".format(len(vocab)))
    print(vocab[:5])
    return vocab


def extract_vocab_params(config):
    model_config = config['model']
    encoder_config = model_config['text_encoder']
    encoder_type = encoder_config['type']
    assert encoder_type == 'ngram'
    ngram_level = encoder_config['ngram_level']
    min_doc_freq = encoder_config['min_doc_freq']
    max_doc_prop = encoder_config['max_doc_prop']
    transform = encoder_config['transform']
    lower = encoder_config['lower']
    digits = encoder_config['convert_digits']
    exclude_nonalpha = encoder_config.get('exclude_nonalpha', False)  # only keep tokens that are just letters (no num or punct)
    require_alpha = encoder_config.get('require_alpha', False)  # only keep tokens that have at least one letter or number
    return ngram_level, min_doc_freq, max_doc_prop, transform, lower, digits, exclude_nonalpha, require_alpha


def convert_to_ngrams(tokens, n, exclude_nonalpha=False, require_alpha=True):
    # convert a list of tokens to n-grams, skipping any that involve tokens without letters or numbers
    if exclude_nonalpha:
        return ['_'.join(tokens[i:i+n]) for i in range(len(tokens)-n+1) if sum([int(t.isalpha()) for t in tokens[i:i+n]]) == n]
    elif require_alpha:
        return ['_'.join(tokens[i:i+n]) for i in range(len(tokens)-n+1) if sum([re.match(r'.*[a-zA-Z0-9]+', t) is None for t in tokens[i:i+n]]) == 0]
    else:
        return ['_'.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

