from assignment_1.src.dataset import Dataset
from decision_tree import *
import nearest_neighbor as knn
import numpy as np


def accuracy(predicted_labels, true_labels):
    """
    Calculates accuracy of predicted labels
    :param predicted_labels: a numpy array of predicted labels
    :param true_labels: a numpy array of true labels
    :return: the accuracy of the predicted labels
    """
    # TODO - implement me
    return np.mean(predicted_labels == true_labels)


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

    knnAccuracies = []
    dtAccuracies = []

    numFolds = len(buckets)

    for foldNumber in range(numFolds):
        trainingData, validationData = get_fold_train_and_validation(buckets, foldNumber)


        predictedKnn = knn.classify_samples(
            labeled_dataset=trainingData,
            unlabeled_samples=validationData.samples,
            k=num_neighbors,
            distance_measure=distance_measure
        )
        knnAccuracy = accuracy(predictedKnn, validationData.labels)
        knnAccuracies.append(knnAccuracy)


        dt = DecisionTree(max_depth=max_depth)
        dt.train(trainingData)


        predictedDt = dt.predict(validationData)

        dtAccuracy = accuracy(predictedDt, validationData.labels)
        dtAccuracies.append(dtAccuracy)

    avgKnnAccuracy = np.mean(knnAccuracies)
    avgDtAccuracy = np.mean(dtAccuracies)
    return avgKnnAccuracy, avgDtAccuracy


if __name__ == '__main__':
    # hard coded parameters
    num_folds = 5

    # load breast cancer data from file
    raw_data = Dataset.load_from_file(os.path.join("../data", "breast-cancer-wisconsin.csv"))

    # shuffle data before split
    raw_data.shuffle()

    # Perform a test/train split
    test_data, training_data = raw_data.test_train_split(20)
    thresholds = find_features_for_continuous_data(training_data)
    training_data = featurize_continuous_data(training_data, thresholds)
    test_data = featurize_continuous_data(test_data, thresholds)
    # create each bucket
    buckets = training_data.split_into_folds(num_folds)
    print(buckets)

    # TODO - implement methods in marked by TODOs in dataset.py
    #  These will be used for cross-validation

    # TODO - implement a hyper-parameter search using average accuracy over
    #  all folds of cross validation as the evaluation metric.
    #  For KNN, check k=1 to 100 with euclidean and cosine distance
    #  For decision trees, check max_depth = 1 to 100

    # TODO - evaluate the effectiveness of each classifier on the test set and
    #  generate any other values needed to fill in the assignment 1 report

    best_knn_accuracy = 0.0
    best_knn_k = None
    best_knn_distance = None

    best_tree_accuracy = 0.0
    best_tree_depth = None

    # ---- KNN Hyper-parameter search ----
    possible_distances = [knn.euclidean, knn.cosine]
    for distance_measure in possible_distances:
        for k in range(1, 20):  # K from 1 to 100
            avg_knn_acc, _ = perform_cross_validation(
                buckets,
                num_neighbors=k,
                distance_measure=distance_measure,
                max_depth=9999999  # large depth so tree doesn't matter here
            )

            if avg_knn_acc > best_knn_accuracy:
                best_knn_accuracy = avg_knn_acc
                best_knn_k = k
                best_knn_distance = distance_measure

    for depth in range(1, 20):
        _, avg_tree_acc = perform_cross_validation(
            buckets,
            num_neighbors=1,
            distance_measure=knn.euclidean,
            max_depth=depth
        )

        if avg_tree_acc > best_tree_accuracy:
            best_tree_accuracy = avg_tree_acc
            best_tree_depth = depth

    print(f"Best KNN Accuracy: {best_knn_accuracy:.3f}, with k={best_knn_k} and distance={best_knn_distance.__name__}")
    print(f"Best Decision Tree Accuracy: {best_tree_accuracy:.3f}, with max_depth={best_tree_depth}")

    print("\nEvaluating best KNN on test set...")
    test_knn_predictions = knn.classify_samples(training_data, test_data.samples,
                                                best_knn_k, best_knn_distance)
    test_knn_accuracy = accuracy(test_knn_predictions, test_data.labels)
    print(f"Test set accuracy (KNN): {test_knn_accuracy:.3f}")

    print("\nbest Decision Tree on test set")
    dt = DecisionTree(max_depth=best_tree_depth)
    dt.train(training_data)
    test_dt_predictions = dt.predict(test_data)
    test_dt_accuracy = accuracy(test_dt_predictions, test_data.labels)
    print(f"Test set accuracy (Decision Tree): {test_dt_accuracy:.3f}")








