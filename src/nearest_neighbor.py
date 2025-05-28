import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from dataset import *


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

    distanceLabelPairs = []
    for label, sample in zip(dataset.labels, dataset.samples):
        dist = distance_measure(sample, new_sample)
        distanceLabelPairs.append((dist, label))


    distanceLabelPairs.sort(key=lambda x: x[0])

    return distanceLabelPairs[0][1]


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
        # Loops through the number of unlabeled samples and appends labels of recently classified samples to labels list
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

    #Iterate over data set and rank by distance create a dict of distance and then the data lable
    # Sort keys by distance
    # Varaible # 0 and # 1
    # Iterate over keys k times change count variables
    #Find max count 1 or 0
    #Return label

    #This avoids collisions even though we didnt have to worry about it
    distanceLabelPairs = []
    # Loops through both labels and samples of the labeled dataset
    for label, sample in zip(labeled_dataset.labels, labeled_dataset.samples):
        #Calculates distance measure of already classified sample and new sample
        dist = distance_measure(sample, new_sample)
        #Appends labeled pairs to distanceLabelPairs
        distanceLabelPairs.append((dist, label))

    # This sorts the first value of the labeled pairs (keys)
    distanceLabelPairs.sort(key=lambda x: x[0])

    # One line list for loop to loop through distance pairs and takes first k labels (k closest in sorted list)
    kLabels = [label for (_, label) in distanceLabelPairs[:k]]

    #Turns list into numpy array with type int
    kLabelsArray = np.array(kLabels, dtype=int)
    #Counts number of each value in the numpy array
    labelCounts = np.bincount(kLabelsArray)
    #Predicted label is the max of the label count which is the most prevalent label
    predictedLabel = np.argmax(labelCounts)

    return predictedLabel



def euclidean(a, b):
    """
    computes the euclidean distance between samples a and b
    :param a: a numpy array representing a sample
    :param b: a numpy array representing a sample
    :return: the euclidean distance between the two samples
    """
    # TODO - implement me

    return np.linalg.norm(a-b)


def cosine(a, b):
    """
    computes the cosine distance between samples a and b
    :param a: a numpy array representing a sample
    :param b: a numpy array representing a sample
    :return: the cosine distance between the two samples
    """
    # TODO - implement me
    return 1 - np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))


if __name__ == '__main__':
    # hard-coded parameters
    distance_measure = euclidean
    k = 5

    # the figure title and output
    fig_output = os.path.join("../output", "KNN_K5_euclidean")
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
    plt.plot(predicted1_samples[:, 0], predicted1_samples[:, 1], 'b*',alpha=0.5 ,label='predicted class a')
    plt.plot(predicted2_samples[:, 0], predicted2_samples[:, 1], 'r*', alpha=0.5, label='predicated class b')
    plt.xlabel('X-axis')
    plt.ylabel('Y-Axis')
    plt.title(fig_title)
    plt.tight_layout()
    plt.grid(True, lw=0.5)
    plt.legend()
    fig.savefig(fig_output)
