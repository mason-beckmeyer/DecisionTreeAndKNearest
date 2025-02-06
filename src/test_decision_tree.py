# Hardcode a small dataset, manually classify it by hand using a decision tree
# Upload a photo of your manual classifications
# Check that your code is correct.
#    Be sure to check Entropy is correctly calculated, and _find_feature_which_best_splits is correct, and that
#    the classifications are what you manually calculated
from assignment_1.src.dataset import Dataset
from assignment_1.src.decision_tree import DecisionTree

samples = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]
labels = [0,0,1,1]
tiny_dataset = Dataset(samples, labels)

tree = DecisionTree()
tree.train(tiny_dataset)
predictions = tree.predict(tiny_dataset)

print("Predictions:", predictions)