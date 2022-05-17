import os
import re
import joblib
from collections import Counter
from optparse import OptionParser

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression as sklearnLogisticRegression
from sklearn.exceptions import NotFittedError

from linear import file_handling as fh
from linear.docs import load_data, encode_documents_as_bow
from linear.labels import build_label_vocab, encode_labels
from linear.vocab import build_vocab
from linear.evaluation import evaluate, check_improvement, compute_proportions


class LogisticRegression:
    """
    Wrapper for LogisticRegression with some extra functionality:
    - allow for a default prediction when only one training class if given
    - return the full matrix of probabilities/coefficients even when certain classes are missing from training data
    Note: This assumes that the class labels will be the integers [0, ..., n_classes - 1]
    """

    def __init__(self, n_classes, C=1.0, penalty='l2', fit_intercept=True, solver='saga', max_iter=100):
        """
        Create a model
        :param n_classes (int): required; number of classes in full label set (not just in training data)
        :param C: default regularization strength; override with set_alpha_values or create_alpha_grid
        :param penalty: regularization type
        :param fit_intercept: If True, fit an intercept in the LR model; otherwise don't
        """
        self._model_type = None
        self._n_classes = n_classes
        self._C = C
        self._penalty = penalty
        self._fit_intercept = fit_intercept
        self._solver = solver
        self._max_iter = max_iter
        self._model = None
        self._default_prediction = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, train_X, train_labels, train_weights=None):
        """
        Fit a logistic regression model to data
        :param train_X: a (sparse) matrix of values
        :param train_labels: a vector of categorical labels
        :param train_weights: a vector of instance weights
        """

        # check to make sure we have been given integer labels
        try:
            assert np.all([isinstance(label, np.int64) for label in train_labels])
        except Exception:
            print(type(train_labels[0]))
            raise ValueError("Train labels are not integers")

        # check to make sure the labels are all within the valid range
        try:
            assert np.all([0 <= label < self._n_classes] for label in train_labels)
        except Exception:
            raise ValueError("Train labels not in range [0, ..., n_classes-1]")

        bincount = np.bincount(train_labels, minlength=self._n_classes)
        most_common = np.argmax(bincount)
        n_instances, n_features = train_X.shape

        # check to see if there is only one label in the training data:
        if bincount[most_common] == len(train_labels):
            print("Only label %d found in training data" % most_common)
            self._model_type = 'default'
            self._default_prediction = most_common
            self._model = None

            # set coef_ and intercept_ to sensible values to match expectations from scikit-learn
            if self._n_classes == 2:
                self.coef_ = np.zeros((1, n_features))
                if most_common == 1:
                    self.intercept_ = np.array(1.0)
                else:
                    self.intercept_ = np.array(-1.0)
            else:
                self.coef_ = np.zeros((self._n_classes, n_features))
                self.intercept_ = np.zeros(self._n_classes)
                self.intercept_[most_common] = 1.0
        else:
            self._model_type = 'LogisticRegression'
            self._model = sklearnLogisticRegression(penalty=self._penalty, C=self._C, fit_intercept=self._fit_intercept, solver=self._solver, max_iter=self._max_iter)
            self._model.fit(train_X, train_labels, sample_weight=train_weights)
            self.coef_ = self._model.coef_
            if self._fit_intercept:
                self.intercept_ = self._model.intercept_

            # set coef_ and intercept_ to match expectations but include all classes
            if self._n_classes == 2:
                self.coef_ = self._model.coef_
                if self._fit_intercept:
                    self.intercept_ = self._model.intercept_
            else:
                self.coef_ = np.zeros((self._n_classes, n_features))
                if self._fit_intercept:
                    self.intercept_ = np.zeros(self._n_classes)
                for i, cl in enumerate(self._model.classes_):
                    self.coef_[cl, :] = self._model.coef_[i, :]
                    if self._fit_intercept:
                        self.intercept_[cl] = self._model.intercept_[i]

    def predict(self, X):
        # if we've stored a default value, then that is our prediction
        if self._model_type == 'default':
            # else, get the model to make predictions
            n_items, _ = X.shape
            return np.ones(n_items, dtype=int) * self._default_prediction
        elif self._model_type == 'LogisticRegression':
            return self._model.predict(X)
        else:
            raise NotFittedError('This LogisticRegression instance is not fitted yet')

    def predict_proba(self, X):
        n_items, _ = X.shape
        full_probs = np.zeros([n_items, self._n_classes])
        # if we've saved a default label, predict that with 100% confidence
        if self._model_type == 'default':
            full_probs[:, self._default_prediction] = 1.0
        elif self._model_type == 'LogisticRegression':
            # otherwise, get probabilities from the model
            model_probs = self._model.predict_proba(X)
            # map the classes that were present in the training data back to the full set of classes
            for i, cl in enumerate(self._model.classes_):
                full_probs[:, cl] = model_probs[:, i]
        else:
            raise NotFittedError('This LogisticRegression instance is not fitted yet')

        return full_probs

    def decision_function(self, X):
        if self._model is None:
            n, p = X.shape
            return np.zeros(n)
        else:
            return self._model.decision_function(X)

    def get_model_size(self):
        if self._model is None:
            return 0
        else:
            coefs = self._model.coef_
            n_nonzero = len(coefs.nonzero()[0])
            return n_nonzero

    def get_n_classes(self):
        return self._n_classes


def get_config_prototype():
    prototype = {
        "seed": 42,
        "dataset_reader": {
            "tokens_field_name": "tokens",
            "label_field_name": None,
            "weight_field_name": None,
            "metadata_dict_name": None,
        },
        "partition_path": "",
        "output_dir": "",
        "model": {
            "type": "linear",
            "penalty": "l1",
            "fit_intercept": True,
            "text_encoder": {
                "type": "ngram",
                "ngram_level": 2,
                "lower": True,
                "min_doc_freq": 1,
                "exclude_nonalpha": False,
                "require_alpha": False,
                "transform": "binarize",
                "stopwords_file": None,
            },
        },
        "trainer": {
            "validation_metric": "f1",
            "average": "micro",
            "n_random_dev_folds": 5,
            "min_alpha": 0.001,
            "max_alpha": 1000,
            "n_alphas": 11,
            "solver": "lbfgs",
            "max_iter": 100
        }
    }
    return prototype




def run(config_file):

    config = fh.read_json(config_file)
    output_dir = config['output_dir']
    partition_file = config['partition_path']

    seed = config.get('seed')
    if seed is not None:
        np.random.seed(int(seed))

    print("Loading data")
    train_docs, dev_docs, test_docs = load_data(partition_file)
    nontest_docs = train_docs + dev_docs

    # combine text fields if desired
    dataset_reader = config["dataset_reader"]
    print("Using tokens field:", dataset_reader['tokens_field_name'])
    print("Using label field:", dataset_reader['label_field_name'])

    print("Building vocab")
    vocab = build_vocab(nontest_docs, config)
    fh.write_to_json(vocab, os.path.join(output_dir, 'vocab.json'), sort_keys=False)

    label_vocab = build_label_vocab(train_docs + dev_docs + test_docs, config)
    n_labels = len(label_vocab)
    fh.write_to_json(label_vocab, os.path.join(output_dir, 'labels.json'), sort_keys=False)

    # encode train and dev docs together
    nontest_ids, nontest_line_indices, nontest_counts, nontest_idf, nontest_weights = encode_documents_as_bow(nontest_docs, vocab, config)
    n_nontest, _ = nontest_counts.shape
    nontest_labels, nontest_weights, nontest_doc_indices = encode_labels(nontest_docs, label_vocab, config)
    if len(nontest_doc_indices) > n_nontest:
        nontest_counts = nontest_counts[nontest_doc_indices, :]
    print("Non-test:", nontest_counts.shape, nontest_labels.shape, nontest_weights.shape)
    if n_labels == 2:
        # take the least common label as the positive class (for computing F1)
        pos_label = int(np.argmin(nontest_labels.sum(0)))
        print("Using {:s} as the positive label".format(str(label_vocab[pos_label])))
    else:
        pos_label = None

    print("Encoding documents and labels")
    train_ids, train_line_indices, train_counts, train_idf, train_weights = encode_documents_as_bow(train_docs, vocab, config, nontest_idf)
    n_train, _ = train_counts.shape
    train_labels, train_weights, train_doc_indices = encode_labels(train_docs, label_vocab, config)

    if len(train_doc_indices) > n_train:
        train_counts = train_counts[train_doc_indices, :]
    print("Train:", train_counts.shape, train_labels.shape, train_weights.shape, train_weights.mean())

    if len(dev_docs) > 0:
        dev_ids, dev_line_indices, dev_counts, _, dev_weights = encode_documents_as_bow(dev_docs, vocab, config, nontest_idf)
        n_dev, _ = dev_counts.shape
        dev_labels, dev_weights, dev_doc_indices = encode_labels(dev_docs, label_vocab, config)
        if len(dev_doc_indices) > n_dev:
            dev_counts = dev_counts[dev_doc_indices, :]
        print("Dev:", dev_counts.shape, dev_labels.shape, dev_weights.shape)
    else:
        dev_counts = None
        dev_labels = None
        dev_weights = None
        dev_ids = None
        dev_line_indices = None

    if len(test_docs) > 0:
        test_ids, test_line_indices, test_counts, _, test_weights = encode_documents_as_bow(test_docs, vocab, config, nontest_idf)
        n_test, _ = test_counts.shape
        test_labels, test_weights, test_doc_indices = encode_labels(test_docs, label_vocab, config)
        if len(test_doc_indices) > n_test:
            test_counts = test_counts[test_doc_indices, :]
        print("Test:", test_counts.shape, test_labels.shape, test_weights.shape)
    else:
        test_counts = None
        test_labels = None
        test_weights = None
        test_ids = None
        test_line_indices = None

    train_label_sums = train_labels.sum(0)
    print("Train label proportions:", train_label_sums / train_label_sums.sum())

    # save train, dev, and test data
    if config['save_data']:
        sparse.save_npz(os.path.join(output_dir, 'train.X.npz'), train_counts)
        np.savez(os.path.join(output_dir, 'train.y.npz'), labels=train_labels)
        if dev_counts is not None:
            sparse.save_npz(os.path.join(output_dir, 'dev.X.npz'), dev_counts)
            np.savez(os.path.join(output_dir, 'dev.y.npz'), labels=dev_labels)
        if test_counts is not None:
            sparse.save_npz(os.path.join(output_dir, 'test.X.npz'), test_counts)
            np.savez(os.path.join(output_dir, 'test.y.npz'), labels=test_labels)

    col_sums = np.array(train_counts.sum(axis=0))
    col_sums = col_sums.reshape((col_sums.size, ))
    start = 0

    vocab_sums = col_sums[start:len(vocab)]
    order = list(np.argsort(vocab_sums))
    order.reverse()
    print("Most common words:".format(' '.join([vocab[i] for i in order[:10]])))
    start += len(vocab)

    print("Training")
    print("Validation metric:", config['trainer']['validation_metric'])
    solver = config['trainer']['solver']
    max_iter = config['trainer']['max_iter']
    models, dev_results, best_alpha_values = train(config, train_counts, train_labels, train_weights, dev_counts, dev_labels, dev_weights, label_vocab, vocab, pos_label=pos_label, output_dir=output_dir, solver=solver, max_iter=max_iter)

    print("Saving everything")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_proportions = compute_proportions(train_labels, train_weights)
    fh.write_to_json(dict(zip(label_vocab, train_proportions)), os.path.join(output_dir, 'proportions.train.json'), sort_keys=False)
    if test_labels is not None:
        test_proportions = compute_proportions(test_labels, test_weights)
        fh.write_to_json(dict(zip(label_vocab, test_proportions)), os.path.join(output_dir, 'proportions.test.json'), sort_keys=False)

    for m_i, model in enumerate(models):
        if model is not None:
            joblib.dump(model, os.path.join(output_dir, 'model' + str(m_i) + '.pkl'))
            fh.write_to_json(dev_results[m_i], os.path.join(output_dir, 'results.dev' + str(m_i) + '.json'))

    print("Doing final evaluation")
    average = config['trainer']['average']
    predict(models, train_counts, train_labels, train_weights, train_ids, train_line_indices, label_vocab, output_dir, 'train', pos_label=pos_label, average=average)
    if dev_counts is not None:
        predict(models, dev_counts, dev_labels, dev_weights, dev_ids, dev_line_indices, label_vocab, output_dir, 'dev', pos_label=pos_label, average=average)
    if test_counts is not None:
        predict(models, test_counts, test_labels, test_weights, test_ids, test_line_indices, label_vocab, output_dir, 'test', pos_label=pos_label, average=average)

    # Train a final model with the average of the best alphas
    best_alpha_val = np.mean(best_alpha_values)
    print("\nTraining a model on all training data with alpha = {:4f}".format(best_alpha_val))
    best_model, _, _ = train(config, nontest_counts, nontest_labels, nontest_weights, dev_counts=None, dev_label_matrix=None, dev_weights=None, label_vocab=label_vocab, vocab=vocab, pos_label=pos_label, output_dir=output_dir, alpha_values=[best_alpha_val], do_random_dev_split=False, solver=solver, max_iter=max_iter)

    with open(os.path.join(output_dir, 'best_alpha_value.txt'), 'w') as f:
        f.write(str(best_alpha_val))

    print("Model with average alpha:")
    if test_counts is not None:
        predict(best_model, test_counts, test_labels, test_weights, test_ids, test_line_indices, label_vocab, output_dir, 'avg.test', pos_label=pos_label, average=average)

    joblib.dump(best_model[0], os.path.join(output_dir, 'model.nontest.pkl'))

    if config['path_for_prediction'] is not None:
        print("Loading data for prediction")
        to_pred_docs = fh.read_jsonlist(config['path_for_prediction'])
        for d_i, doc in enumerate(to_pred_docs):
            doc['_i'] = 'pr_' + str(d_i)
            doc['_i'] = 'pr_' + str(d_i)
        to_pred_ids, to_pred_line_indices, to_pred_counts, _, to_pred_weights = encode_documents_as_bow(to_pred_docs, vocab, config, nontest_idf)
        n_to_pred, _ = to_pred_counts.shape
        try:
            to_pred_labels, to_pred_weights, to_pred_doc_indices = encode_labels(to_pred_docs, label_vocab, config)
            if len(to_pred_doc_indices) > n_to_pred:
                print("Size mismsatch in to_predict")
                to_pred_counts = to_pred_counts[to_pred_doc_indices, :]
            print("To predict:", to_pred_counts.shape, to_pred_labels.shape, to_pred_weights.shape)
            print("Pred:", to_pred_counts.shape)
            predict(models, to_pred_counts, to_pred_labels, to_pred_weights, to_pred_ids, to_pred_line_indices, label_vocab, output_dir, 'to_predict', pos_label=pos_label, average=average, do_evaluation=True)
            predict(best_model, to_pred_counts, to_pred_labels, to_pred_weights, to_pred_ids, to_pred_line_indices, label_vocab, output_dir, 'avg.to_predict', pos_label=pos_label, average=average, do_evaluation=True)
        except Exception as e:
            print("Raising exception", e, "; ignoring to pred labels")
            if n_labels == 2:
                to_pred_labels = np.zeros((n_to_pred, ))
            else:
                to_pred_labels = np.zeros((n_to_pred, n_labels))
            print("Pred:", to_pred_counts.shape)
            predict(models, to_pred_counts, to_pred_labels, to_pred_weights, to_pred_ids, to_pred_line_indices, label_vocab, output_dir, 'to_predict', pos_label=pos_label, average=average, do_evaluation=False)
            predict(best_model, to_pred_counts, to_pred_labels, to_pred_weights, to_pred_ids, to_pred_line_indices, label_vocab, output_dir, 'avg.to_predict', pos_label=pos_label, average=average, do_evaluation=False)


def train(config, train_counts, train_label_matrix, train_weights=None, dev_counts=None, dev_label_matrix=None, dev_weights=None, label_vocab=None, vocab=None, pos_label=1, output_dir=None, alpha_values=None, do_random_dev_split=True, solver='lbfgs', max_iter=100):
    model_config = config['model']
    model_type = model_config['type']
    assert model_type == 'linear'
    penalty = model_config['penalty']

    trainer = config['trainer']
    metric = trainer['validation_metric']
    average = trainer['average']
    n_random_dev_folds = trainer.get('n_random_dev_folds')
    fit_intercept = model_config['fit_intercept']

    n_train, n_classes = train_label_matrix.shape

    order = np.arange(n_train)
    if dev_counts is None:
        if do_random_dev_split:
            n_dev_folds = n_random_dev_folds
            np.random.shuffle(order)
        else:
            n_dev_folds = 1
    else:
        n_dev_folds = 1
    fold_size = int(n_train // n_dev_folds)

    if alpha_values is None:
        alpha_values = create_alpha_grid(config)

    models = []
    best_alpha_values = []
    dev_results_all = []
    for f in range(n_dev_folds):
        if n_dev_folds > 1:
            if f < n_dev_folds - 1:
                dev_indices = np.array(order[f * fold_size:(f+1) * fold_size])
            else:
                dev_indices = np.array(order[f * fold_size:])
            train_indices = np.array(list(set(np.arange(n_train)) - set(dev_indices)))

            print("Fold {:d}: splitting training data into ".format(f), len(train_indices), len(dev_indices))

            model, dev_results, best_alpha_val = train_one_model(alpha_values, metric, average, penalty, train_counts[train_indices, :], train_label_matrix[train_indices, :], train_weights[train_indices], dev_counts=train_counts[dev_indices, :], dev_label_matrix=train_label_matrix[dev_indices, :], dev_weights=train_weights[dev_indices], pos_label=pos_label, fit_intercept=fit_intercept, solver=solver, max_iter=max_iter)
            dev_sums = np.array(train_counts[dev_indices, :].sum(axis=0)).reshape((len(vocab), ))
        else:
            model, dev_results, best_alpha_val = train_one_model(alpha_values, metric, average, penalty, train_counts, train_label_matrix, train_weights, dev_counts, dev_label_matrix, dev_weights, pos_label=pos_label, fit_intercept=fit_intercept, solver=solver, max_iter=max_iter)
            if dev_counts is None:
                dev_sums = None
            else:
                dev_sums = np.array(dev_counts.sum(axis=0)).reshape((len(vocab), ))

        best_alpha_values.append(best_alpha_val)
        models.append(model)
        dev_results_all.append(dev_results)
        #if dev_sums is not None:
        #    dev_freqs = dev_sums / np.sum(dev_sums)

        # print top terms in the best model
        if label_vocab is not None and vocab is not None:
            all_coefs = model.coef_

            if output_dir is not None:
                with open(os.path.join(output_dir, 'top_terms.txt'), 'w') as f:
                    f.write('\n')

            if n_classes == 2:
                print("Top weighted features:")
                coef_order = list(np.argsort(all_coefs[0]))
                words = [vocab[i] for i in coef_order[:20] if all_coefs[0][i] < 0]
                print(label_vocab[0], ':', ' '.join(words))
                write_words_to_file(output_dir, 'top_terms.txt', label_vocab[0], words)

                coef_order.reverse()
                words = [vocab[i] for i in coef_order[:20] if all_coefs[0][i] > 0]
                print(label_vocab[1], ':', ' '.join(words))
                write_words_to_file(output_dir, 'top_terms.txt', label_vocab[1], words)

                if dev_sums is not None:
                    print("Highest impact features:")
                    coef_order = list(np.argsort(all_coefs[0] * np.log(dev_sums+1)))
                    words = [vocab[i] for i in coef_order[:20] if all_coefs[0][i] < 0]
                    print(label_vocab[0], ':', ' '.join(words))
                    write_words_to_file(output_dir, 'top_terms.txt', str(label_vocab[0]) + ' weighted', words)

                    coef_order.reverse()
                    words = [vocab[i] for i in coef_order[:20] if all_coefs[0][i] > 0]
                    print(label_vocab[1], ':', ' '.join(words))
                    write_words_to_file(output_dir, 'top_terms.txt', str(label_vocab[1]) + ' weighted', words)

            else:
                print("Top weighted features:")
                for label_i, label in enumerate(label_vocab):
                    coef_order = list(np.argsort(all_coefs[label_i]))
                    coef_order.reverse()
                    words = [vocab[i] for i in coef_order[:20] if all_coefs[label_i][i] > 0]
                    print(label, ':', ' '.join(words))
                    write_words_to_file(output_dir, 'top_terms.txt', label, words)

                if dev_sums is not None:
                    print("Highest impact features:")
                    for label_i, label in enumerate(label_vocab):
                        coef_order_weighted = list(np.argsort(all_coefs[label_i] * np.log(dev_sums+1)))
                        coef_order_weighted.reverse()
                        words = [vocab[i] for i in coef_order_weighted[:20] if all_coefs[label_i][i] > 0]
                        print(label, ':', ' '.join(words))
                        write_words_to_file(output_dir, 'top_terms.txt', str(label) + ' weighted', words)

                if config['dataset_reader']['feda'] is not None:
                    print("\nFEDA:")
                    print("Top weighted features:")
                    for label_i, label in enumerate(label_vocab):
                        coef_order = np.argsort(all_coefs[label_i])[::-1]
                        words = [vocab[i] for i in coef_order if all_coefs[label_i][i] > 0 and '__' not in vocab[i]]
                        print(label, ':', ' '.join(words[:20]))

                    if dev_sums is not None:
                        print("Highest impact features:")
                        for label_i, label in enumerate(label_vocab):
                            coef_order_weighted = np.argsort(all_coefs[label_i] * np.log(dev_sums+1))[::-1]
                            words = [vocab[i] for i in coef_order_weighted if all_coefs[label_i][i] > 0 and '__' not in vocab[i]]
                            print(label, ':', ' '.join(words[:20]))

    return models, dev_results_all, best_alpha_values


def train_one_model(alpha_values, metric, average, penalty, train_counts, train_label_matrix, train_weights=None, dev_counts=None, dev_label_matrix=None, dev_weights=None, pos_label=1, fit_intercept=True, solver='lbfgs', max_iter=100):
    train_labels = np.argmax(train_label_matrix, axis=1)

    models = []
    pred_probs_list = []
    best_model = None
    best_metric_val = None
    best_alpha_index = 0
    dev_pred_probs = None
    best_pred_probs = None
    n_train, n_classes = train_label_matrix.shape

    for i, alpha in enumerate(alpha_values):
        model = LogisticRegression(n_classes=n_classes, penalty=penalty, C=alpha, fit_intercept=fit_intercept, solver=solver, max_iter=max_iter)
        model.fit(train_counts, train_labels, train_weights)
        train_pred_probs = model.predict_proba(train_counts)
        train_eval = evaluate(train_label_matrix, train_pred_probs, metric, train_weights, pos_label=pos_label, average=average)
        if dev_counts is not None:
            dev_pred_probs = model.predict_proba(dev_counts)
            dev_eval = evaluate(dev_label_matrix, dev_pred_probs, metric, dev_weights, pos_label=pos_label, average=average)
        else:
            dev_eval = -1
        size = model.get_model_size()
        print("alpha={:0.4f}, {:s}: train={:0.4f} dev={:0.4f} size={:d}".format(alpha, metric, train_eval, dev_eval, size))
        if best_metric_val is None:
            best_metric_val = dev_eval
            best_model = model
            best_pred_probs = dev_pred_probs
        elif check_improvement(best_metric_val, dev_eval, metric):
            best_metric_val = dev_eval
            best_model = model
            best_alpha_index = i
            best_pred_probs = dev_pred_probs

    best_alpha_val = alpha_values[best_alpha_index]

    metrics = ['f1', 'accuracy', 'mae', 'calibration']
    dev_results = {}
    if dev_counts is not None:
        for m in metrics:
            dev_results[m] = evaluate(dev_label_matrix, best_pred_probs, m, dev_weights, pos_label=pos_label, average=average)
    print("Best alpha = {:0.4f}".format(best_alpha_val), dev_results, '\n')
    return best_model, dev_results, best_alpha_val


def create_alpha_grid(config):
    # create a grid of alpha values evenly spaced in log-space
    trainer = config['trainer']

    min_alpha = trainer['min_alpha']
    max_alpha = trainer['max_alpha']
    n_alphas = trainer['n_alphas']

    if n_alphas > 1:
        alpha_factor = np.power(max_alpha / min_alpha, 1.0/(n_alphas-1))
        alpha_values = np.array(min_alpha * np.power(alpha_factor, np.arange(n_alphas)))
    else:
        alpha_values = np.array([min_alpha])

    return alpha_values


def predict(models, counts, labels, weights, ids, line_indices, label_vocab, output_dir, output_prefix='test', pos_label=1, average='micro', do_evaluation=True):

    if counts is not None:
        metrics = ['f1', 'accuracy', 'mae', 'calibration']
        test_results = {}
        pred_probs = []
        for model in models:
            pred_probs.append(model.predict_proba(counts))

        pred_probs = np.mean(pred_probs, axis=0)
        if do_evaluation and labels is not None:
            for m in metrics:
                test_results[m] = evaluate(labels, pred_probs, m, weights, pos_label=pos_label, average=average)

            test_pred_proportions = compute_proportions(pred_probs, weights)
            fh.write_to_json(dict(zip(label_vocab, test_pred_proportions)), os.path.join(output_dir, 'proportions.' + output_prefix + '.pred.json'), sort_keys=False)

            print(output_prefix + ":", test_results)
            fh.write_to_json(test_results, os.path.join(output_dir, 'results.' + output_prefix + '.json'))

            _, n_classes = pred_probs.shape
            if n_classes > 2:
                print("Label\tTrue %\tPred %\tHard %\tACC\tF1\tMAE")
                for label_i, label in enumerate(label_vocab):
                    acc = evaluate(labels[:, label_i], np.array(np.argmax(pred_probs, axis=1) == label_i, dtype=int), 'accuracy', weights, one_dim=True, average=average)
                    f1 = evaluate(labels[:, label_i], np.array(np.argmax(pred_probs, axis=1) == label_i, dtype=int), 'f1', weights, one_dim=True, pos_label=1, average=average)
                    mae = evaluate(labels[:, label_i], pred_probs[:, label_i], 'mae', weights, one_dim=True, average=average)
                    label_percent = np.dot(labels[:, label_i], weights) / weights.sum()
                    pred_percent = np.dot(pred_probs[:, label_i], weights) / weights.sum()
                    hard_percent = np.dot((pred_probs[:, label_i] >= 0.5), weights) / weights.sum()
                    if type(label) == str:
                        print("{:s}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(label[:10], label_percent, pred_percent, hard_percent, acc, f1, mae))
                    else:
                        print("{:d}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(label, label_percent, pred_percent, hard_percent, acc, f1, mae))

        # also save predicted probs
        df = pd.DataFrame(pred_probs, index=line_indices, columns=label_vocab)
        df.to_csv(os.path.join(output_dir, 'pred.probs.' + output_prefix + '.csv'))

        df = pd.DataFrame(ids, index=line_indices)
        df.to_csv(os.path.join(output_dir, 'ids.' + output_prefix + '.csv'))


def write_words_to_file(output_dir, filename, label, words):
    if output_dir is not None:
        with open(os.path.join(output_dir, filename), 'a') as f:
            f.write(str(label) + '\n')
            for word in words:
                f.write(word + '\n')
            f.write('\n')


def make_config(input_file,
                exp_dir,
                name='linear',
                metric='f1',
                tokens_field_name='tokens',
                split_text=False,
                macro=False,
                transform='binarize',
                ngram_level=2,
                min_df=2,
                max_dp=1.0,
                stopwords_file=None,
                feda=None,
                min_alpha=0.01,
                max_alpha=1000.,
                n_alphas=11,
                convert_digits=False,
                exclude_nonalpha=False,
                require_alphanum=False,
                penalty='l1',
                no_intercept=False,
                weight_field_name=None,
                solver='liblinear',
                max_iter=100,
                pred_file=None,
                save_data=False):

    config = get_config_prototype()

    partition = fh.read_json(input_file)
    label_field_name = partition['task']

    config['dataset_reader']['tokens_field_name'] = tokens_field_name
    config['dataset_reader']['split_text'] = split_text
    config['dataset_reader']['label_field_name'] = label_field_name
    config['dataset_reader']['feda'] = feda
    config['dataset_reader']['weight_field_name'] = weight_field_name

    name = name + '_' + metric
    if transform is not None:
        name += '_' + transform
    name += '_n' + str(ngram_level)
    name += '_' + penalty
    if weight_field_name is not None:
        name += '_weighted'
    if convert_digits:
        name += '_digits'
    if feda is not None:
        name += '_feda_' + str(feda)

    output_dir = os.path.join(exp_dir, name)
    config['output_dir'] = output_dir
    config['partition_path'] = input_file
    config['path_for_prediction'] = pred_file
    config['save_data'] = save_data
    config['trainer']['validation_metric'] = metric
    if macro:
        config['trainer']['average'] = 'macro'
    else:
        config['trainer']['average'] = 'micro'
    config['trainer']['min_alpha'] = min_alpha
    config['trainer']['max_alpha'] = max_alpha
    config['trainer']['n_alphas'] = n_alphas
    config['trainer']['solver'] = solver
    config['trainer']['max_iter'] = max_iter
    config['model']['text_encoder']['transform'] = transform
    config['model']['text_encoder']['ngram_level'] = ngram_level
    #config['model']['text_encoder']['min_ngram_level'] = ngram_level
    config['model']['text_encoder']['lower'] = True
    config['model']['text_encoder']['min_doc_freq'] = min_df
    config['model']['text_encoder']['max_doc_prop'] = max_dp
    config['model']['text_encoder']['convert_digits'] = convert_digits
    config['model']['text_encoder']['exclude_nonalpha'] = exclude_nonalpha
    config['model']['text_encoder']['require_alpha'] = require_alphanum
    config['model']['text_encoder']['stopwords_file'] = stopwords_file
    config['model']['penalty'] = penalty
    if no_intercept:
        config['model']['fit_intercept'] = False
    else:
        config['model']['fit_intercept'] = True

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, 'config.json')
    print("Saving config to {:s}".format(output_path))
    fh.write_to_json(config, output_path)
    return output_path


def main():
    usage = "%prog parition.jsonlist"
    parser = OptionParser(usage=usage)
    parser.add_option('--name', dest='name', default='linear',
                      help='Name for model / output directory: default=%default')
    parser.add_option('-t', dest='tokens_field_name', default='tokens',
                      help='Text field name (assumes a list of lists (tokenized sents)): default=%default')
    parser.add_option('--split-text', action="store_true", default=False,
                      help='Instead do simple white space splitting of text in tokens_field_name: default=%default')
    parser.add_option('--metric', dest='metric', default='f1',
                      help='Metric [f1|accuracy|calibration|calibration_old]: default=%default')
    parser.add_option('--macro', action="store_true", dest="macro", default=False,
                      help='Use macro average for f1: default=%default')
    parser.add_option('--transform', dest='transform', default='binarize',
                      help='Transform [None|binarize|tfidf]: default=%default')
    parser.add_option('-n', type=int, dest='ngram_level', default=2,
                      help='n-gram level: default=%default')
    #parser.add_option('--min-n', type=int, default=1,
    #                  help='Minimum n-gram level: default=%default')
    parser.add_option('--min-df', type=int, default=2,
                      help='Minimum document frequency for vocab: default=%default')
    parser.add_option('--max-dp', type=float, default=1.0,
                      help='Maximum document frequency for vocab: default=%default')
    parser.add_option('--stopwords-file', type=str, default=None,
                      help='.txt file with a list of stopwords (one per line): default=%default')
    parser.add_option('--feda', type=str, default=None,
                      help='Field to use for domain adaptation: default=%default')
    #parser.add_option('--use-dev', action="store_true", dest="use_dev", default=False,
    #                  help='Used the given dev fold (otherwise recombine and ensemble): default=%default')
    parser.add_option('--min-alpha', type=float, default=0.01,
                      help='Minimum alpha value: default=%default')
    parser.add_option('--max-alpha', type=float, default=1000,
                      help='Maximum alpha value: default=%default')
    parser.add_option('--n-alphas', type=float, default=11,
                      help='Number of alpha values: default=%default')
    parser.add_option('--digits', action="store_true", default=False,
                      help='Convert digits to #: default=%default')
    parser.add_option('--exclude-nonalpha', action="store_true", default=False,
                      help='Exclude all tokens except those made up of only letters: default=%default')
    parser.add_option('--require-alphanum', action="store_true", default=False,
                      help='Only include tokens with at least one letter or number: default=%default')
    parser.add_option('--penalty', type=str, default='l1',
                      help='Regularization type [l1|l2]: default=%default')
    parser.add_option('--no-intercept', action="store_true", default=False,
                      help="Don't fit an intercept: default=%default")
    parser.add_option('--weight-field', type=str, default=None,
                      help='Name of weight field (if any): default=%default')
    parser.add_option('--solver', type=str, default='liblinear',
                      help='Solver [liblinear|sag|saga|lbfgs]: default=%default')
    parser.add_option('--max-iter', type=int, default=100,
                      help='Maximum iterations for solver: default=%default')
    parser.add_option('--run', action="store_true", dest="run", default=False,
                      help='Run the experiment after making the config: default=%default')
    parser.add_option('--pred-file', type=str, default=None,
                      help='File to make predictions on: default=%default')
    parser.add_option('--save-data', action="store_true", default=False,
                      help='Save encoded data and labels: default=%default')

    (options, args) = parser.parse_args()

    input_file = args[0]
    exp_dir = os.path.split(input_file)[0]

    name = options.name
    metric = options.metric
    if metric != 'f1' and metric != 'accuracy' and metric != 'calibration':
        raise ValueError('Metric not recognized')
    tokens_field_name = options.tokens_field_name
    split_text = options.split_text
    macro = options.macro
    transform = options.transform
    ngram_level = options.ngram_level
    #min_ngram_level = options.min_n
    min_df = options.min_df
    max_dp = options.max_dp
    stopwords_file = options.stopwords_file
    convert_digits = options.digits
    exclude_nonalpha = options.exclude_nonalpha
    require_alphanum = options.require_alphanum
    penalty = options.penalty
    weight_field_name = options.weight_field
    pred_file = options.pred_file
    save_data = options.save_data

    autorun = options.run

    config_file = make_config(input_file=input_file,
                              exp_dir=exp_dir,
                              name=name,
                              metric=metric,
                              tokens_field_name=tokens_field_name,
                              split_text=split_text,
                              macro=macro,
                              transform=transform,
                              ngram_level=ngram_level,
                              min_df=min_df,
                              max_dp=max_dp,
                              stopwords_file=stopwords_file,
                              feda=options.feda,
                              min_alpha=options.min_alpha,
                              max_alpha=options.max_alpha,
                              n_alphas=options.n_alphas,
                              convert_digits=convert_digits,
                              exclude_nonalpha=exclude_nonalpha,
                              require_alphanum=require_alphanum,
                              penalty=penalty,
                              no_intercept=options.no_intercept,
                              weight_field_name=weight_field_name,
                              solver=options.solver,
                              max_iter=options.max_iter,
                              pred_file=pred_file,
                              save_data=save_data, )

    if autorun:
        run(config_file)


if __name__ == '__main__':
    main()
