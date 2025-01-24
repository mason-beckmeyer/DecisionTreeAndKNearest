**PCSE 595 – Spring 2022
Assignment 1**


The goal of this assignment is to gain an understanding of some machine learning fundamentals such as loading, handling, and visualizing data, get some practice with common Python packages, ensure you are familiar with two basic classifiers, and understand proper experimental set up.

You will code two different classifiers, ensure their correctness on an easy dataset, and apply them to a more complex dataset. You will tune their hyperparameters and compare the two using proper experimental set up. 

Specific requirements:
1.	Implement a K Nearest Neighbors Classifier
	a.	Implement Euclidean and cosine distance metrics
	b. 	Design a simple test case and verify the correctness of your algorithm
	b.	The KNN classifier has “k” and the distance metric as hyperparameters
	d.	Follow instructions in nearest_neighbor.py

2.	Implement a Decision Tree Classifier
	a.	Featurize the data for use in a decision tree by converting continuous numeric data to binary data by finding borders between classes 
	b.	Use random and information gain as splitting criteria
	c.	Design a simple test case and verify the correctness of your algorithm
	d.	Follow instructions in decision_tree.py
	e.	Manually check your code and follow the instructions in test_decision_tree.py

3.	Compare the performance of the two algorithms on the Wisconsin Breast Cancer Dataset using proper experimental set up
	a.	Use a cross validation to tune the hyperparameters
	b.	Use a test dataset to evaluate performance
	c.	Use accuracy as your evaluation metric
	d.	Implement TODOs in dataset.py
	e.	Follow instructions in run_experiment.py

To reduce the workload, I provided some base code. You should fill in relevant sections marked with TODO statements. All your code must be from scratch. You may not use pre-existing libraries to do the work for you. The ONLY libraries you may use are already imported into the project code. These libraries include: os, csv, sys, numpy, matplotlib, random.

All work must be your own. You may not show anyone your code. You may not share your code with anyone. You may discuss general concepts with your classmates. You may not use pre-existing implementations on the internet. Use the class slides as your first reference, use the provided textbooks as your second reference, and Dr. Henry as your third reference. 

Focus on understanding the material. Searching the internet will likely confuse you. The best way to implement machine learning is to choose a single implementation strategy (i.e. the one in the slides) and implement it based on the math. There are many differences in derivations and implementation details of machine learning algorithms, and a heavy reliance on pre-existing packages (which you aren’t allowed to use). I have given you everything you need to implement the algorithms – use it.

**Dataset Details**

Wisconsin Breast Cancer Dataset:
This dataset is provided to you in data/breast-cancer-wisconsin.csv. 
The dataset consists of 699 samples with 9 features each. The features all have values between 1 and 10. The features are:

0. Clump Thickness
1. Uniformity of Cell Size
2. Uniformity of Cell Shape
3. Marginal Adhesion
4. Single Epithelial Cell Size
5. Bare Nuclei
6. Bland Chromatin
7. Normal Nucleoli
8. Mitoses

Each row in data/breast-cancer-wisconsin.csv corresponds to a single sample, each column a feature, and the label is provided in the last column. A label of 0 indicates benign (non-cancerous), and a label of 1 indicates malignant (cancerous). 

**Deliverables**

Submit all deliverables via gitlab. These should include:

1)     Photos of complete worksheets (covariance calculation, Distance Measures and Nearest Neighbors Classifier, and Nearest Centroid Classifier using Mahalanobis Distance)
2)	All of your code
3)	test_decision_tree.py is completed, and the photo upload of your manual calculations
4)	The figure output from nearest_neighbors.py with your name in the title
	a.	Use Euclidean distance and k = 5 to generate the figure
5)	The figure output from decision_tree.py with your name in the title
	a.	Use a max_depth of 10 to generate the figure
6)	A completed assignment1_report.docx
	a.	Fill in the blanks for the template provided and answer the questions at the bottom
	b.	Please leave all of your inserted text highlighted (it makes it easier for me to grade)
