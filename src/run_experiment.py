from src.decision_tree import *
import src.nearest_neighbor as knn
import numpy as np


def accuracy(predicted_labels, true_labels):
    """
    Calculates accuracy of predicted labels
    :param predicted_labels: a numpy array of predicted labels
    :param true_labels: a numpy array of true labels
    :return: the accuracy of the predicted labels
    """
    # TODO - implement me
    pass


def perform_cross_validation(buckets, num_neighbors=1, distance_measure=knn.euclidean, max_depth=9999999):
    """
    Performs cross validation on a knn classifier and a decision tree classifier using
    the specified hyper-parameters

    :param buckets: the divided training dataset
    :param num_neighbors: the number of neighbors in knn
    :param distance_measure: the distance mesure for knn
    :param max_depth: the maximum depth for decision tree
    :return: the average validation accuracy over all folds for the
             knn classifier and decision tree respectively
    """
    # TODO - implement me
    pass


if __name__ == '__main__':
    # hard coded parameters
    num_folds = 5

    # load breast cancer data from file
    raw_data = Dataset.load_from_file(os.path.join("data", "breast-cancer-wisconsin.csv"))

    # shuffle data before split
    raw_data.shuffle()

    # Perform a test/train split
    test_data, training_data = raw_data.test_train_split(20)

    # create each bucket
    buckets = training_data.split_into_folds(num_folds)

    # TODO - implement methods in marked by TODOs in dataset.py
    #  These will be used for cross-validation

    # TODO - implement a hyper-parameter search using average accuracy over
    #  all folds of cross validation as the evaluation metric.
    #  For KNN, check k=1 to 100 with euclidean and cosine distance
    #  For decision trees, check max_depth = 1 to 100

    # TODO - evaluate the effectiveness of each classifier on the test set and
    #  generate any other values needed to fill in the assignment 1 report




