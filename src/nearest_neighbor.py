import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from src.dataset import *


def nearest_neighbor(dataset, new_sample, distance_measure):
    """
    Classifies the sample using the label of the single nearest neighbor

    :param dataset: the labeled dataset to classify the unlabled sample
    :param new_sample: a numpy array representing the sample to classify
    :param distance_measure: the distance measure to use
    :return: the label of new_sample
    """
    # TODO - implement me first - make sure your code works with a single
    #  nearest neighbor before adapting it to k-nearest neighbors
    pass


def classify_samples(labeled_dataset, unlabeled_samples, k, distance_measure):
    """
    Classifies the samples using k nearest neighbors classification

    :param labeled_dataset: the labeled dataset used to classify the unlabeled samples
    :param unlabeled_samples: a numpy array of samples to classify. Each
                              row corresponds to a sample, each column a feature
    :param k: the number of neighbors to vote on the classification
    :param distance_measure: the distance measure to use
    :return: a numpy array of labels for each unlabeled sample
    """
    labels = []
    for i in range(unlabeled_samples.shape[0]):
        labels.append(classify_sample(labeled_dataset, unlabeled_samples[i, :], k, distance_measure))
    return np.array(labels)


def classify_sample(labeled_dataset, new_sample, k, distance_measure):
    """
    Classifies the sample using k nearest neighbors classification

    :param labeled_dataset: the labeled dataset used to classify the sample
    :param new_sample: a numpy array representing the sample to classify
    :param k: the number of neighbors to vote on the classification
    :param distance_measure: the distance measure to use
    :return: the label of new_sample
    """
    # TODO - implement me
    #  Hint: this can be trickier than it seems, find the distance to all points
    #        then, find the k-closest points, then take a vote using those points
    pass


def euclidean(a, b):
    """
    computes the euclidean distance between samples a and b
    :param a: a numpy array representing a sample
    :param b: a numpy array representing a sample
    :return: the euclidean distance between the two samples
    """
    # TODO - implement me
    pass


def cosine(a, b):
    """
    computes the cosine distance between samples a and b
    :param a: a numpy array representing a sample
    :param b: a numpy array representing a sample
    :return: the cosine distance between the two samples
    """
    # TODO - implement me
    pass


if __name__ == '__main__':
    # hard-coded parameters
    distance_measure = euclidean
    k = 5

    # the figure title and output
    fig_output = os.path.join("output", "KNN_K5_euclidean")
    fig_title = 'KNN (K=5) Euclidean Distance Classification'

    # generate test and training data
    data1 = generate_data(500, [1, 1], [[0.3, 0.2], [0.2, 0.2]], 0)
    data2 = generate_data(500, [3, 4], [[0.3, 0], [0, 0.2]], 1)
    dataset = combine_data(data1, data2)
    unlabeled_data = generate_data(300, [2, 3], [[0.3, 0.0], [0.0, 0.2]], -1)

    # perform knn
    unlabeled_data.labels = classify_samples(dataset, unlabeled_data.samples, k, distance_measure)

    # split the data by class - for plotting
    predicted1_samples = unlabeled_data.samples[(unlabeled_data.labels == 0), :]
    predicted2_samples = unlabeled_data.samples[(unlabeled_data.labels == 1), :]

    # plot the samples
    fig = plt.figure()
    plt.plot(data1.samples[:, 0], data1.samples[:, 1], 'b.', label='given class a')
    plt.plot(data2.samples[:, 0], data2.samples[:, 1], 'r.', label='given class b')
    plt.plot(predicted1_samples[:, 0], predicted1_samples[:, 1], 'g*', label='predicted class a')
    plt.plot(predicted2_samples[:, 0], predicted2_samples[:, 1], '*', color='orange', label='predicated class b')
    plt.xlabel('X-axis')
    plt.ylabel('Y-Axis')
    plt.title(fig_title)
    plt.tight_layout()
    plt.grid(True, lw=0.5)
    plt.legend()
    fig.savefig(fig_output)
