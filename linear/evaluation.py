import numpy as np
from sklearn.metrics import f1_score


def evaluate(true_label_matrix, pred_prob_matrix, metric, weights, one_dim=False, pos_label=1, average='micro'):
    if metric == 'accuracy':
        if one_dim:
            value = np.dot(true_label_matrix == pred_prob_matrix, weights) / weights.sum()
        else:
            value = np.dot(np.argmax(true_label_matrix, axis=1) == np.argmax(pred_prob_matrix, axis=1), weights) / weights.sum()
    elif metric == 'f1':
        if one_dim:
            value = weighted_f1(true_label_matrix, pred_prob_matrix, n_classes=2, pos_label=pos_label, weights=weights, average='binary')
        else:
            n_items, n_classes = pred_prob_matrix.shape
            value = weighted_f1(np.argmax(true_label_matrix, axis=1), np.argmax(pred_prob_matrix, axis=1), n_classes=n_classes, pos_label=pos_label, average=average, weights=weights)
    elif metric == 'calibration':
        values = calibration_score(true_label_matrix, pred_prob_matrix, weights)
        value = np.mean(values)
    elif metric == 'calibration_new':
        value = calibration_score_new(true_label_matrix, pred_prob_matrix, weights)
    elif metric == 'mae':
        value = compute_proportions_mae(true_label_matrix, pred_prob_matrix, weights)
    else:
        raise ValueError("Metric not recognized.")
    return value


def weighted_f1(true, pred, n_classes=2, pos_label=1, average='micro', weights=None):
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
        else:
            f1 = f1_score(true, pred, average='binary', labels=range(n_classes), pos_label=pos_label, sample_weight=weights)
    else:
        if average is None:
            f1 = f1_score(true, pred, average='micro', labels=range(n_classes), pos_label=None, sample_weight=weights)
        else:
            f1 = f1_score(true, pred, average=average, labels=range(n_classes), pos_label=None, sample_weight=weights)
    return f1


def calibration_score(label_matrix, pred_probs_matrix, weights, n_bins=None):
    """
    assume binary data
    binary_label_vector: a vector of binary labels
    pred_probs: a matrix of predicted probabilities of the positive class [n x Y]
    weights: a vector of weights
    """

    n_items, n_classes = pred_probs_matrix.shape
    assert n_items == len(label_matrix)

    if n_bins is None:
        n_bins = int(n_items // 50)+1

    # for binary problems, both results should be the same, so just consider one
    if n_classes == 2:
        n_classes = 1

    # otherwise we will average over classes
    calibration_per_class = np.zeros(n_classes)

    # compute the calibration per class
    for c in range(n_classes):
        pred_probs = pred_probs_matrix[:, c]

        # sort the predicted probabilities for this class
        order = np.argsort(pred_probs)

        breakpoints = list(np.array(np.arange(n_bins)/float(n_bins) * n_items, dtype=int).tolist()) + [n_items]

        mae = 0.0
        for b in range(n_bins):
            start = breakpoints[b]
            end = breakpoints[b+1]
            indices = order[start:end]
            mean_bin_probs = np.dot(pred_probs[indices], weights[indices]) / np.sum(weights[indices])
            mean_bin_labels = np.dot(label_matrix[indices, c], weights[indices]) / np.sum(weights[indices])
            ae = np.abs(mean_bin_labels - mean_bin_probs)
            mae += ae

        calibration_per_class[c] = mae / n_bins

    return calibration_per_class


def compute_proportions_mae(true_label_matrix, pred_prob_matrix, weights=None):
    true_label_proportions = compute_proportions(true_label_matrix, weights)
    pred_prob_proportions = compute_proportions(pred_prob_matrix, weights)
    mae = np.mean(np.abs(true_label_proportions - pred_prob_proportions))
    return mae


def compute_proportions(pred_prob_matrix, weights=None):
    n_items = len(pred_prob_matrix)
    if weights is None:
        weights = np.ones(n_items)
    proportions = np.dot(weights, pred_prob_matrix) / weights.sum()
    return proportions


def check_improvement(old_val, new_val, metric):
    if metric == 'accuracy':
        return new_val > old_val
    elif metric == 'f1':
        return new_val > old_val
    elif metric == 'calibration':
        return new_val < old_val
    elif metric == 'mae':
        return new_val < old_val
    else:
        print("Metric not recognized")
        sys.exit()


def evaluate_multilabel(true_label_matrix, pred_prob_matrix, metric, weights, average='micro'):
    n_items, n_labels = true_label_matrix.shape
    if metric == 'accuracy':
        value = np.dot(np.sum(np.array(true_label_matrix, dtype=int) == np.array(pred_prob_matrix >= 0.5, dtype=int), axis=1), weights) / (weights.sum() * n_labels)
    elif metric == 'f1':
        value = weighted_f1(true_label_matrix, np.array(pred_prob_matrix >= 0.5, dtype=int), n_classes=2, pos_label=1, average=average, weights=weights)
    else:
        raise ValueError("Metric not recognized.")

    return value


def calibration_score_new(label_matrix, pred_probs_matrix, weights, n_bins=None):

    n_items, n_classes = pred_probs_matrix.shape
    assert n_items == len(label_matrix)

    if n_classes == 2:
        return calibration_score_binary(label_matrix, pred_probs_matrix, weights, n_bins)
    else:
        return calibration_score_multiclass(label_matrix, pred_probs_matrix, weights, n_bins)


def calibration_score_binary(label_matrix, pred_probs_matrix, weights, n_bins=None):

    n_items, n_classes = pred_probs_matrix.shape
    if n_bins is None:
        n_bins = int(n_items // 50)

    order = np.argsort(pred_probs_matrix[:, 1])
    # make a list of indices indicating how we will split the data into bins
    breakpoints = list(np.array(np.arange(n_bins)/float(n_bins) * n_items, dtype=int).tolist()) + [n_items]

    mae = 0.0
    for b in range(n_bins):
        start = breakpoints[b]
        end = breakpoints[b+1]
        indices = order[start:end]
        mean_bin_probs = np.dot(pred_probs_matrix[indices, 1], weights[indices]) / np.sum(weights[indices])
        mean_bin_labels = np.dot(label_matrix[indices, 1], weights[indices]) / np.sum(weights[indices])
        ae = np.abs(mean_bin_labels - mean_bin_probs)
        mae += ae

    return mae / n_bins


def calibration_score_multiclass(label_matrix, pred_probs_matrix, weights, min_bins=None):

    n_items, n_classes = pred_probs_matrix.shape

    # count the number of items that would be predicted for each class
    predictions = np.argmax(pred_probs_matrix, axis=1)
    pred_class_counts = np.bincount(predictions, weights=weights, minlength=n_classes)
    #print(pred_class_counts)

    if min_bins is None:
        # try to get bins with a decent number of instances, but also more than the number of classes (heuristic)
        min_bins = max(int(n_items // 50), int(np.sum(pred_class_counts > 0) * 1.5))
    #print(min_bins)

    # create a number of bins per class in proportion to the number that would be predicted
    bins_per_class = np.array(pred_class_counts, dtype=float) / float(min_bins)
    #print(bins_per_class)
    # take the ceiling to make sure there is at least one bin for classes with at least one instance
    bins_per_class = np.array(np.ceil(bins_per_class), dtype=int)
    #print(bins_per_class)

    # compute an average of the MAE over all bins for each class
    mae = 0.0
    for cl in range(n_classes):
        if bins_per_class[cl] > 0:
            # break the points predicted to be in this class into bins along the gradation of the class probabilities
            bins_cl = bins_per_class[cl]
            class_indices = predictions == cl
            pred_probs_cl = pred_probs_matrix[class_indices, :]
            label_matrix_cl = label_matrix[class_indices, :]
            weights_cl = weights[class_indices]
            order = np.argsort(pred_probs_cl[:, cl])
            n_items_cl = int(class_indices.sum())

            breakpoints = list(np.array(np.arange(bins_cl)/float(bins_cl) * n_items_cl, dtype=int).tolist()) + [n_items_cl]

            class_mae = 0.0
            for b in range(bins_cl):
                start = breakpoints[b]
                end = breakpoints[b+1]
                bin_indices = order[start:end]
                # take the mae over the full vector of probabilities and class labels
                mean_bin_probs = np.dot(weights_cl[bin_indices], pred_probs_cl[bin_indices, :]) / np.sum(weights_cl[bin_indices])
                mean_bin_labels = np.dot(weights_cl[bin_indices], label_matrix_cl[bin_indices, :]) / np.sum(weights_cl[bin_indices])
                ae = np.mean(np.abs(mean_bin_labels - mean_bin_probs))
                #print(cl, b, ae)
                class_mae += ae

            mae += class_mae

    return mae / bins_per_class.sum()

