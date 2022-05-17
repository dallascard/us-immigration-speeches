import numpy as np


def build_label_vocab(documents, config):
    vocab = set()

    dataset_reader = config["dataset_reader"]
    label_field_name = dataset_reader['label_field_name']

    for doc in documents:
        label = doc[label_field_name]
        if type(label) == int:
            vocab.add(label)
        elif type(label) == str:
            vocab.add(label)
        elif type(label) == bool:
            vocab.add(label)
        else:
            raise ValueError("label field type not accepted")

    vocab = list(vocab)
    vocab.sort()
    print("Number of classes = {:d}".format(len(vocab)))
    print(vocab)
    return vocab


def encode_labels(documents, label_vocab, config):
    n_classes = len(label_vocab)
    label_index = dict(zip(label_vocab, range(n_classes)))

    dataset_reader = config["dataset_reader"]
    label_field_name = dataset_reader['label_field_name']
    weight_field_name = dataset_reader.get('weight_field_name')

    labels = []
    indices = []
    weights = []

    for i, doc in enumerate(documents):
        # if we see a label that's not in the vocabulary (e.g. in dev/test data), just treat it as having no label
        label_field = doc[label_field_name]

        if label_field is not None:
            doc_labels = np.zeros(n_classes, dtype=int)
            doc_labels[label_index[label_field]] = 1
            labels.append(doc_labels)
            indices.append(i)
            if weight_field_name in doc:
                weights.append(doc[weight_field_name])
            else:
                weights.append(1.0)

    labels = np.vstack(labels)
    indices = np.array(indices, dtype=int)
    weights = np.array(weights, dtype=float)

    return labels, weights, indices
