This homework was coded in python.

The corresponding files are:
	hw5.py - Contains example code for running each classifier
	DecisionTree.py - Our decision tree implementation
	RandomForest.py - Our random forest implementation
	AdaBoost.py - Our adaboost implementation
	utils.py - Cross validation function and entropy functions 
	test.py - Random test code

The following libraries where used:
	numpy - Used for all array/matrix computations
	pylab - Used for plotting
	scipy - Used to load the matlab files into python
	sklearn - Used for the cross validation function
	ipdb - Used for debugging

Our classifiers auto train on initializations.
To classify a dataset, just call the Classifier.classify() function.
Refer to hw5.py for examples on how to run each classifier.

Desciption of each classifier:
	Decision Tree:
		The naive implementation with no pruning and no depth limiting.

	Random Forest:
		Trains M different trees. Each tree is constructed with a random number of points and a random number of features.

	AdaBoost:
		Instead of rewriting each classifier to account for weights, we just sample the dataset using the weighted distribution.
