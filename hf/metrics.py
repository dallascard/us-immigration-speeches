from collections import Counter

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix, classification_report

import numpy as np


def accuracy(preds, labels, weights=None):
    if weights is None:
        return simple_accuracy(preds, labels)
    else:
        return weighted_accuracy(preds, labels, weights)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def weighted_accuracy(preds, labels, weights):
    return (weights * (preds == labels)).sum() / weights.sum()


def weighted_f1(pred, true, weights=None, n_classes=2, pos_label=1, average='micro'):
    """
    Override f1_score in sklearn in order to deal with both binary and multiclass cases
    :param true: true labels
    :param pred: predicted labels
    :param n_classes: total number of different possible labels
    :param pos_label: label to use as the positive label for the binary case (0 or 1)
    :param average: how to calculate f1 for the multiclass case (default = 'micro')

    :return: f1 score
    """

    if n_classes == 2:
        if np.sum(true * pred) == 0:
            f1 = 0.0
        elif average is None:
            f1 = f1_score(true, pred, average='binary', labels=range(n_classes), pos_label=pos_label, sample_weight=weights)
        elif average == 'macro':
            f1 = f1_score(true, pred, average='macro', labels=range(n_classes), pos_label=pos_label, sample_weight=weights)
        else:
            f1 = f1_score(true, pred, average='binary', labels=range(n_classes), pos_label=None, sample_weight=weights)
    else:
        if average is None:
            f1 = f1_score(true, pred, average='micro', labels=range(n_classes), pos_label=None, sample_weight=weights)
        else:
            f1 = f1_score(true, pred, average=average, labels=range(n_classes), pos_label=None, sample_weight=weights)
    return f1


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    micro_f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
    macro_f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "micro_f1": micro_f1,
        "macro_f1":macro_f1,
        "acc_and_macro_f1": (acc + macro_f1) / 2,
    }


def cm(preds, true, labels=None, sample_weights=None):
    return confusion_matrix(preds, true, labels=labels, sample_weight=None)


def calibration(pred_probs, true_probs, sample_weights=None, n_bins=10, decomposition='brier', binning='adaptive', epsilon=1e-6):

    # Note: true_probs can be either labels (0/1) or "true" probabilities
    # For the binary case, pred_probs and true_probs should be vectors
    # For the multiclass case, pred_probs and true_probs should be matrices [n_items x n_classes]

    ndim = np.ndim(pred_probs)

    if sample_weights is None:
        sample_weights = np.ones(len(true_probs))

    if decomposition == 'brier':
        assert ndim == 1
        if binning == 'even':
            thresholds = np.arange(n_bins+1)/float(n_bins)
        elif binning == 'adaptive':
            qs = list(100 * np.arange(n_bins+1) / float(n_bins))[1:-1]
            quantiles = list(np.percentile(pred_probs, q=qs))
            thresholds = [0.] + quantiles + [1.01]
        else:
            raise ValueError("Binning not recognized:", binning)
        cal = 0
        for t_i, t in enumerate(thresholds[:-1]):
            indices = np.array(t <= pred_probs < thresholds[t_i+1])
            mean_pred_prob = np.mean(pred_probs[indices])
            mean_true_prob = np.mean(true_probs[indices])
            cal += np.power(mean_true_prob - mean_pred_prob, 2) * sample_weights[indices].sum()
        cal = np.sqrt(cal / sample_weights.sum())

    elif decomposition == 'log':
        n_itmes, n_classes = pred_probs.shape

        cal = 0
        for cl in range(n_classes):
            class_cal = 0
            if binning == 'even':
                thresholds = np.arange(n_bins+1)/float(n_bins)
            elif binning == 'adaptive':
                qs = list(100 * np.arange(n_bins+1) / float(n_bins))[1:-1]
                quantiles = list(np.percentile(pred_probs[:, cl], q=qs))
                print(cl, quantiles)
                thresholds = [0.] + quantiles + [1.01]
            else:
                raise ValueError("Binning not recognized:", binning)
            for t_i, t in enumerate(thresholds[:-1]):
                indices = np.array(t <= pred_probs[:, cl] < thresholds[t_i+1])
                mean_pred_prob = np.mean(pred_probs[indices, cl])
                mean_true_prob = np.mean(true_probs[indices, cl])
                class_cal += -np.log(mean_pred_prob / np.max(epsilon, mean_true_prob)) * mean_true_prob * sample_weights[indices].sum()

            # weight class calibration by number of examples that have this as the true label
            cal += np.sqrt(class_cal / sample_weights.sum()) * np.sum(true_probs[:, cl])
        cal = cal / true_probs.sum()
    else:
        raise ValueError("Decomposition not recognized:", decomposition)

    return cal


def refinement(pred_probs, true_probs=None, sample_weights=None, decomposition='brier', epsilon=1e-6):
    ndim = np.ndim(pred_probs)
    if decomposition == 'brier':
        assert ndim == 1
        if sample_weights is None:
            sample_weights = np.ones(len(pred_probs))
        r = np.sum(sample_weights * (pred_probs * (1 - pred_probs)) / sample_weights.sum())
        rs = [r]
    elif decomposition == 'log':
        if ndim == 1:
            pred_probs = np.vstack([pred_probs, 1-pred_probs]).T
        elif ndim != 2:
            raise ValueError("pred_probs must have dimensions 1 or 2:", ndim)
        n_items, n_classes = pred_probs.shape
        if sample_weights is None:
            sample_weights = np.ones(n_items)
        temp = - pred_probs * np.log(np.maximum(np.ones_like(pred_probs) * epsilon, pred_probs))
        r = np.sum(temp.sum(1) * sample_weights) / sample_weights.sum()
        if true_probs is None:
            rs = [r]
        else:
            rs = np.sum(temp * true_probs * sample_weights.reshape((n_items, 1)), 0) / true_probs.mean(0) / sample_weights.sum()
    else:
        raise ValueError("decomposition not recognized", decomposition)

    return r, rs


def classif_report(preds, labels, target_names):
    return classification_report(labels, preds, labels=[0,1,2], target_names=target_names, output_dict=True)


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(metrics, preds, labels, classes, weights=None, pred_probs=None, true_probs=None, epsilon=1e-6):
    results = {}
    n_classes = len(classes)
    for metric in metrics:
        print(metric, n_classes)
        if metric == 'accuracy':
            results['acc'] = simple_accuracy(preds, labels)
        elif metric == 'weighted_accuracy':
            results['weighted_acc'] = weighted_accuracy(preds, labels, weights)
        elif metric == 'f1':
            results['f1'] = weighted_f1(preds, labels, weights=weights, n_classes=n_classes)
        elif metric == 'weighted_f1':
            results['weighted_f1'] = weighted_f1(preds, labels, weights=weights, n_classes=n_classes)
        elif metric == 'micro_f1':
            results['micro_f1'] = weighted_f1(preds, labels, weights=weights, n_classes=n_classes, average='micro')
        elif metric == 'macro_f1':
            results['macro_f1'] = weighted_f1(preds, labels, weights=weights, n_classes=n_classes, average='macro')
        elif metric == 'per_class_f1':
            per_class_f1 = f1_score(labels, preds, labels=range(n_classes), average=None, sample_weight=weights)
            print(per_class_f1)
            for cl in range(n_classes):
                results['f1-' + classes[cl]] = per_class_f1[cl]
        elif metric == 'cfm':
            results['cfm'] = cm(preds, labels, labels=None, sample_weights=weights)
        elif metric.startswith('calibration'):
            _, decomposition, binning, bins = metric.split('-')
            results['metric'] = calibration(pred_probs, true_probs, sample_weights=weights, n_bins=int(bins),
                                            decomposition=decomposition, binning=binning, epsilon=epsilon)
        elif metric.startswith('refinement'):
            _, decomposition = metric.split('-')
            results['metric'] = refinement(pred_probs, sample_weights=weights, decomposition=decomposition,
                                           epsilon=epsilon)
        else:
            raise ValueError("Metric {:s} not recognized".format(metric))
    return results


def glue_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "hans":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == 'framing':
        return {'acc': simple_accuracy(preds, labels)}
    elif task_name == 'tone' or task_name == 'tone-weighted':
        return {'acc': simple_accuracy(preds, labels),
                'acc_and_f1': acc_and_f1(preds, labels),
                'cm': cm(preds, labels),
                'per_class': classif_report(preds, labels, target_names=['pro', 'neutral', 'anti'])}
    elif task_name == 'relevant' or task_name == 'relevant-weighted':
        return {'acc': simple_accuracy(preds, labels),
                'acc_and_f1': acc_and_f1(preds, labels)}
    elif task_name == 'climate' or task_name == 'climate-weight':
        return {'acc': simple_accuracy(preds, labels),
                'acc_and_f1': acc_and_f1(preds, labels),
                'cm': cm(preds, labels),
                'per_class': classif_report(preds, labels, target_names=["disagree", "neutral", "agree"])}
    elif task_name == 'binary':
        return {'acc': simple_accuracy(preds, labels),
                'acc_and_f1': acc_and_f1(preds, labels)}
    else:
        raise KeyError(task_name)
