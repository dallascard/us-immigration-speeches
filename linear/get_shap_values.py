import re
import json
from optparse import OptionParser

import shap
import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--train-file', type=str, default='data/speeches/Congress/imm_segments_with_tone_labels_43-67_train.jsonlist',
                      help='Train file: default=%default')
    parser.add_option('--test-file', type=str, default='data/speeches/Congress/imm_segments_with_tone_labels_43-67_test.jsonlist',
                      help='Test file: default=%default')
    parser.add_option('--stopwords-file', type=str, default='linear/snowball.txt',
                      help='Stopwords file: default=%default')
    parser.add_option('--n-terms', type=int, default=20,
                      help='Number to print for each class: default=%default')
    parser.add_option('--min-count', type=int, default=20,
                      help='Min count: default=%default')
    #parser.add_option('--log-freqs', action="store_true", default=False,
    #                  help='Log frequencies: default=%default')
    #parser.add_option('--sqrt-freqs', action="store_true", default=False,
    #                  help='Sqrt frequencies: default=%default')
    parser.add_option('--print-vals', action="store_true", default=False,
                      help='Print weights and shapley values: default=%default')

    (options, args) = parser.parse_args()

    train_file = options.train_file
    test_file = options.test_file
    stopwords_file = options.stopwords_file
    min_count = options.min_count
    n_terms = options.n_terms
    print_vals = options.print_vals

    print("Loading data")
    with open(train_file) as f:
        lines = f.readlines()
    train_lines = [json.loads(line) for line in lines]
    print(len(train_lines))

    with open(test_file) as f:
        lines = f.readlines()
    test_lines = [json.loads(line) for line in lines]
    print(len(test_lines))

    with open(stopwords_file) as f:
        lines = f.readlines()
    stopwords = set([line.strip() for line in lines])

    label_map = {'anti': 0, 'neutral': 1, 'pro': 2}

    print("Preprocessing")
    train_corpus = []
    train_y = []
    for line in train_lines:
        #if line['label'] != 'neutral':
        sents = []
        for sent in line['tokens']:
            tokens = [t.lower() for t in sent if t.lower() not in stopwords and t.isalpha()]
            sents.append(' '.join(tokens))
        text = ' '.join(sents)
        train_corpus.append(text)
        train_y.append(label_map[line['label']])

    test_corpus = []
    test_y = []
    for line in test_lines:
        #if line['label'] != 'neutral':    
        sents = []
        for sent in line['tokens']:
            tokens = [t.lower() for t in sent if t.lower() not in stopwords and t.isalpha()]
            sents.append(' '.join(tokens))
        text = ' '.join(sents)
        test_corpus.append(text)
        test_y.append(label_map[line['label']])

    print("Vectorizing")
    vectorizer = TfidfVectorizer(min_df=min_count)
    X_train = vectorizer.fit_transform(train_corpus).toarray() # sparse also works but Explanation slicing is not yet supported
    X_test = vectorizer.transform(test_corpus).toarray()

    print("Fitting model")
    model = sklearn.linear_model.LogisticRegression(penalty="l1", C=1, solver='liblinear')
    model.fit(X_train, train_y)

    print("Getting shap values")
    explainer = shap.Explainer(model, X_train, feature_names=vectorizer.get_feature_names())
    shap_values = explainer(X_test)

    shap_values.shape
    feature_names = vectorizer.get_feature_names()
    coefs = model.coef_

    for label in range(3):
        print(label)
        mean_values = shap_values[:, :, label].mean(0)
        mean_values = np.array(mean_values.values, dtype=float)
        order = np.argsort(mean_values)[::-1]
        count = 0
        i = 0
        while count < n_terms:
            index = order[i]
            if np.argmax(coefs[:, index]) == label:
                if print_vals:
                    print(feature_names[index], mean_values[index], coefs[label][index])
                else:
                    print(feature_names[index])
                count += 1
            i += 1
        print()
    

if __name__ == '__main__':
    main()
