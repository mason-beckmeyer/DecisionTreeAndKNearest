import numpy as np
from dataset import *
import matplotlib.pyplot as plt
import os
import random




def find_features_for_continuous_data(data):
    """
    Finds features for continuous data - features are halfway points
    between class boundaries
giytiuytuyttut
    :param data: the data to create features for
    :return: a list of lists describing the split values for each dimension
             the first dimension contains a list for each dimension of the original data
             each of those lists contains a single list of values which describe where
             to split the data in that dimension. For example, a return value of
             [[1.2,3.7],[2.9]] indicates that 3 features were found. The first feature
             feature asks the question "Is dimension 1 less than 1.2?". The second feature
             asks the question "Is dimension 1 less than 3.5?", and the third feature asks
             the question "Is dimension 2 less than 2.9?"
    """

    # TODO - implement me
    # Hint: In my implementation I created a dictionary of value-label pairs for each feature
    #       which I called points. I sorted the points by their value:
    #            sorted_points = sorted(points.items(), key=lambda item: item[0]))
    #       Then, if there labels changed, I found a feature as the midpoint between the two
    #       values.

    # Minor Implementation note: When creating the dictionary of points there may be collisions
    # where is already in the dictionary (possibly with a different label). It is OK to ignore
    # these collisions (just overwrite the previous value).
    num_features = data.num_features
    features_per_dimension = []

    for dim in range(num_features):
        value_label_map = {}
        for sample, label in zip(data.samples, data.labels):
            #appends the dimension of the sample with the label value into the map
            value_label_map[sample[dim]] = label
        # Sorts the keys by value
        sorted_items = sorted(value_label_map.items(), key=lambda x: x[0])

        thresholds = []

        # Compare adjacent items in sorted order
        for i in range(1, len(sorted_items)):
            prev_val, prev_lab = sorted_items[i - 1]
            curr_val, curr_lab = sorted_items[i]
            if prev_lab != curr_lab:
                midpoint = (prev_val + curr_val) / 2
                thresholds.append(midpoint)
                print("this is the midpoint",midpoint)

        features_per_dimension.append(thresholds)

    return features_per_dimension


def featurize_continuous_data(data, features):
    """
    Featurizes a dataset based on the features described in features

    :param data: a dataset object
    :param features: features generated from find_features_for_continuous_data
    :return: the featurized dataset
    """
    # TODO implement me
    # Hint: Recall, you create a set of features which indicates the feature number
    #       and value for a "is sample[feature_number] < value" type test. This method
    #       converts numeric data to binary (True/False) data using those features

    samples = data.samples
    num_samples = data.num_samples
    #Finds total binary features
    total_binary_features = sum(len(thresh_list) for thresh_list in features)
    # Turns 0s to False and 1s to True
    bin_features = np.zeros((num_samples, total_binary_features), dtype=bool)

    feature_index = 0
    # Avoids nested loops this is shorthand
    for dim, threshold_list in enumerate(features):
        for threshold in threshold_list:
            bin_features[:, feature_index] = samples[:, dim] < threshold
            feature_index += 1
    # Returns dataset with samples of the features and the data labels
    return Dataset(bin_features.astype(int), data.labels)

def entropy(dataset):
    """
    Calculates entropy of a dataset

    :param dataset: a dataset object
    :return: the entropy of the dataset
    """
    # TODO - implement me

    labels = dataset.labels
    totalSamples = len(labels)

    if totalSamples == 0:
        return 0
    # Returns the unique labels no dupes
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    entropyV = -np.sum((label_counts / totalSamples) * np.log2(label_counts / totalSamples))

    return entropyV


class Node:
    def __init__(self, data=None):
        """
        Creates a new node object.

        Each node contains a dataset, and optionally a test and a true and false child node.
        The dataset is stored in the data parameter.
        Recall that data is featurized as True/False features, so the test stored at this node
        checks (if samples[feature_index] == True). The dataset is split on this test, and
        samples which are true for that feature are stored in true_child, and features which
        are false are stored in false child.

        :param data: the data this node contains
        """
        self.data = data
        self.feature_index = None
        self.true_child = None
        self.false_child = None

    def __str__(self):
        """
        converts this node to a string
        :return: a string representation of the node
        """
        return self.data.class_ratio_string() + " split by feature " + str(self.feature_index)

    def split_by(self, feature_index):
        """
        Splits the data contained in this node by the binary feature at feature_index
        :param feature_index: the feature_index to split on
        :return: a tuple of dataset objects consisting of (data_true, data_false) - where
                 data_true contains samples that are True at feature_index and data_false
                 contains samples that are false at data_index
        """
        # TODO - implement me
        #  Hint: This is a helper function for find_feature_which_best_spits.
        #  It splits the dataset at this node into samples which are
        #  true vs. false for the feature at feature_index

        samples = self.data.samples
        labels = self.data.labels

        # Create boolean masks for filtering
        mask_true = samples[:, feature_index] == 1
        mask_false = samples[:, feature_index] == 0


        return Dataset(samples[mask_true], labels[mask_true]), Dataset(samples[mask_false], labels[mask_false])

    def find_feature_which_best_splits(self):
        """
        Finds the feature which best splits the data in this node using
        information gain as the splitting criteria

        :return: the index of the feature that best splits the data
                 returns -1 if no feature increases information
        """
        # TODO - implement me

        best = -1
        bestGain = 0


        pEntropy = entropy(self.data)
        numFeatures = self.data.num_features

        for idx in range(numFeatures):
            dataTrue, dataFalse = self.split_by(idx)

            # Skip empty splits
            if dataTrue.num_samples == 0 or dataFalse.num_samples == 0:
                continue

            leftEntropy = entropy(dataTrue)
            rightEntropy = entropy(dataFalse)

            total = self.data.num_samples
            weighted = ((dataTrue.num_samples / total) * leftEntropy +
                        (dataFalse.num_samples / total) * rightEntropy)

            infoGain = pEntropy - weighted

            if infoGain > bestGain:
                bestGain = infoGain
                best = idx
        print("This is best info gain ",best)
        return best


class DecisionTree:
    def __init__(self, max_depth=100):
        """
        Constructs a new DecisionTree object
        :param max_depth: the maximum depth the DecisionTree will reach during training
        """
        self.root = Node()
        self.max_depth = max_depth
        self.features = None

    def predict(self, data):
        """
        Uses the trained decision tree to predict the class of each
        sample in data
        :param data: the data to classify
        :return: a numpy array of binary labels corresponding to the
                 prediction of each sample in data
        """
        # TODO - implement me
        predictions = [self._predict_sample(sample, self.root) for sample in data.samples]
        return np.array(predictions)

    def _predict_sample(self, sample, current_node):
        """
        Recursive method for predicting the class of a sample.

        :param sample: the sample to predict
        :param current_node: the current node in the decision tree
        :return: True/False indicating the class label of the sample
        """
        # TODO implement me
        #  Hint: to perform prediction, traverse the tree with the sample.
        #  The final prediction should be the majority class of leaf node
        #  reached by traversing the tree with the sample.
        if current_node.true_child is None and current_node.false_child is None:
            return current_node.predicted_class


        if sample[current_node.feature_index]:
            return self._predict_sample(sample, current_node.true_child)
        else:
            return self._predict_sample(sample, current_node.false_child)

    def __str__(self):
        """
        Creates a string representation of this DecisionTree
        :return:
        """
        return self._recursive_string(self.root, "")

    def _recursive_string(self, node, leading_tabs):
        """
        Recursive helper function for __str__
        :param leading_tabs: the number of tabs to indent
        :return: a string representation of this DecisionTree
        """
        string = leading_tabs + str(node)
        if node.true_child is not None:
            string += "\n"
            string += self._recursive_string(node.true_child, leading_tabs + "\t")
            string += "\n"
            string += self._recursive_string(node.false_child, leading_tabs + "\t")
        return string

    def train(self, data):
        """
        Creates a decision tree for classifying the data
        :param data: the data to classify
        :return: None
        """
        self.root.data = data
        self._divide(self.root, 0)

        print("Training Complete")
        print(str(self))

    def _divide(self, current_node, depth):
        """
        Recursive helper method for Train - constructs the decision tree by
        recursively dividing the training data and storing the split decisions
        at nodes in the tree. Remember, datasets are divided based on a sample's
        feature value, NOT the class value

        :param current_node: The current node to be split
        :param depth: the depth of the current node
        :return: None - the constructed decision tree is stored starting with self.root
        """
        # TODO - implement me
        #  Hint: this is the recursive method that is used to train the decision tree.
        #   Remember the base cases, which are:
        #      all classes are the same
        #      splitting by a feature adds no information
        #      depth > max depth


        labels = current_node.data.labels

        if len(set(labels)) == 1:
            current_node.is_leaf = True
            current_node.predicted_class = labels[0]
            return


        if depth >= self.max_depth:
            current_node.is_leaf = True
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            majority_label = unique_labels[np.argmax(label_counts)]
            current_node.predicted_class = majority_label

            return


        bestFeature = current_node.find_feature_which_best_splits()


        if bestFeature == -1:
            current_node.is_leaf = True
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            majority_label = unique_labels[np.argmax(label_counts)]
            current_node.predicted_class = majority_label
            return


        dataTrue, dataFalse = current_node.split_by(bestFeature)



        current_node.feature_index = bestFeature
        current_node.true_child = Node(Dataset(dataTrue.samples, dataTrue.labels))
        current_node.false_child = Node(Dataset(dataFalse.samples, dataFalse.labels))

        self._divide(current_node.true_child, depth + 1)
        self._divide(current_node.false_child, depth + 1)




if __name__ == '__main__':
    # hard-coded parameters
    max_depth = 10
    fig_output = os.path.join("../output", "Decision_Tree_max_depth_10")
    fig_title = 'Decision Tree (Max_Depth=10) Classification [TODO - insert your name here]'

    # generate test and training data
    data1 = generate_data(500, [1, 1], [[0.3, 0.2], [0.2, 0.2]], 0)
    data2 = generate_data(500, [3, 4], [[0.3, 0], [0, 0.2]], 1)
    training_dataset = combine_data(data1, data2)
    test_dataset = generate_data(300, [2, 3], [[0.3, 0.0], [0.0, 0.2]], -1)

    # find features in the training data
    features = find_features_for_continuous_data(training_dataset)
    feature_index = 0
    for dimension_number, dimension_features in enumerate(features):
        for val in dimension_features:
            print("feature " + str(feature_index) + ": is dimension " + str(dimension_number) + " < " + str(val))
            feature_index += 1

    # featurize the training and test dataset
    featurized_training_dataset = featurize_continuous_data(training_dataset, features)
    featurized_test_dataset = featurize_continuous_data(test_dataset, features)

    # create a Decision Tree object
    sample_tree = DecisionTree()
    sample_tree.train(featurized_training_dataset)
    featurized_test_dataset.labels = sample_tree.predict(featurized_test_dataset)

    # split the data by class - for plotting (use the non-featurized data to plot)
    predicted1_samples = test_dataset.samples[(featurized_test_dataset.labels == 0), :]
    predicted2_samples = test_dataset.samples[(featurized_test_dataset.labels == 1), :]

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
