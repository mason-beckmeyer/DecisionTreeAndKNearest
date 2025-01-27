from math import remainder

import numpy as np
import csv


def get_fold_train_and_validation(buckets, fold_num):
    """
    Gets training and validation sets for a fold of cross-validation

    :param buckets: the divided dataset
    :param fold_num: the fold number
    :return: a tuple of Dataset objects corresponding to the fold
             training data and fold validation data respectively
    """
    # create training data for this fold
    fold_training_samples = np.ones([0, buckets[0].num_features])
    fold_training_labels = np.ones([0])
    for i in range(len(buckets)):
        if i != fold_num:
            fold_training_samples = np.concatenate([fold_training_samples, buckets[i].samples])
            fold_training_labels = np.concatenate([fold_training_labels, buckets[i].labels])
    fold_training_data = Dataset(fold_training_samples, fold_training_labels)

    # create validation data for this fold
    validation_data = buckets[fold_num]

    return fold_training_data, validation_data


def combine_data(data1, data2):
    """
    Combines two datasets into a single dataset

    :param data1: a dataset
    :param data2: another dataset
    :return: a dataset consisting of data1 and data2 combined
    """
    data = np.concatenate((data1.samples, data2.samples))
    labels = np.concatenate((data1.labels, data2.labels))

    return Dataset(data, labels)


def generate_data(n, mean, cov, label):
    """
    generates a dataset with a multi-variate gaussian distribution

    :param n: the number of samples to generate
    :param mean: the mean of the distribution
    :param cov: the covariance of the distribution
    :param label: the label for each generated sample
    :return: a dataset
    """
    data = np.random.multivariate_normal(mean, cov, n)
    labels = np.zeros([n]) + label
    return Dataset(data, labels)


class Dataset:
    def __init__(self, samples, labels):
        """
        Constructs a new Dataset object

        :param samples: a sample number by feature table-like object
                        of samples (e.g. a NumPy array or list of lists)
        :param labels: a list-like object of labels (e.g. a NumPy array or list of labels)
        """
        # set the samples and labels
        self.samples = np.array(samples)
        self.labels = np.array(labels)

        # determine the number of samples and features in the dataset
        self.num_samples = 0
        self.num_features = 0
        if len(self.samples.shape) == 2:
            self.num_samples, self.num_features = self.samples.shape

    def load_from_file(file_path):
        """
        Loads data from a .csv file

        The file should contain a sample per line, features are comma seperated
        The last line should contain the class

        :return: a Dataset object of the data in the file
        """
        samples = []
        labels = []
        with open(file_path, 'rt') as file:
            rows = csv.reader(file, delimiter=',')

            # read in each sample
            for row in rows:
                # ensure all values are read in as floats
                row_vals = []
                for val in row:
                    row_vals.append(float(val))

                # save the sample and label
                samples.append(row_vals[:-1])
                labels.append(row_vals[-1])
        return Dataset(samples, labels)

        # loading data is super easy with pandas, but you aren't allowed
        # to use pandas for this assignment
        # import pandas as pd
        # df = pd.read_csv(file_path, header=None)
        # data = df.to_numpy()
        # return Dataset(data[:, :-1], data[:, -1])

    def all_classes_same(self):
        """
        Checks if all classes are the same in the dataset
        :return: True if all classes are the same, False otherwise
        """
        # if there are no labels, then all the classes are the same
        if len(self.labels) == 0:
            return True
        # there are labels, so check if they are all the same
        for label in self.labels:
            if label != self.labels[0]:
                return False
        return True

    def class_ratio_string(self):
        """
        Returns a string indicating the number of positive and number negative samples
        :return: a string representing the class ratio
        """
        num_samples = self.samples.shape[0]
        num_pos = sum(self.labels)
        num_neg = num_samples - num_pos
        return str(num_pos) + " pos, " + str(num_neg) + " neg"

    def get_majority_class(self):
        """
        Determines the majority class in the dataset
        :return: True if True is the majority class; False otherwise
        """
        true_count = 0
        false_count = 0
        for label in self.labels:
            if label:
                true_count += 1
            else:
                false_count += 1
        return true_count > false_count

    def shuffle(self):
        """
        Randomly reorders this data
        :return: None
        """
        shuffle = np.random.permutation(range(0, self.num_samples))
        self.labels = self.labels[shuffle]
        self.samples = self.samples[shuffle, :]

    def test_train_split(self, test_percent):
        """
        Splits the first test_percent of the data as the test set, and saves
        the remaining as the training set.

        :param test_percent: the percentage of the data to reserve as a test dataset
                             Note: a value of 20, not 0.2 should be given to split 20%
                             of the data into the test set. Not
        :return: a tuple containing two Dataset Objects (test_dataset, training_dataset)
        """
        numIndex = int(len(self.labels)*(test_percent/100))

        numberTestSamples,numberTestLabels = self.labels[0:numIndex],self.labels[0:numIndex]
        numberTrainingSamples,numberTrainingLabels = self.labels[numIndex:],self.labels[numIndex:]

        testDataSet = Dataset(numberTestSamples,numberTestLabels)
        trainingDataSet = Dataset(numberTrainingSamples,numberTrainingLabels)

        return  (testDataSet,trainingDataSet)






    def split_into_folds(self, num_folds):
        """
        Divides the data into folds and returns a list of Dataset objects
        which contains each split of the data. The dataset objects need to
        combined each fold to properly form the training and validation
        datasets for that fold using get_fold_train_and_validation

        For example, this for five folds this method will split a dataset
        into 5 different datasets. For fold 1 of cross validation, dataset 1
        can serve as the validation set, and datasets 2,3,4,5 should be
        combined to serve as the training dataset for that fold

        :param num_folds: the number of folds for cross-validation
        :return: a list of Dataset objects containing a Dataset per split
        """

        num, remainder = divmod(len(self.samples), num_folds)

        startIndex = 0

        result = []

        for i in range(num_folds):
            end = startIndex + num + (1 if i < remainder else 0)

            result.append(Dataset(self.samples[startIndex:end],self.labels[startIndex:end]))

            startIndex = end

        return result
