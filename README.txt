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
	sklearn - Used for part of our cross validation function, and for part of our pruning.
	ipdb - Used for debugging


Put the spam.mat file into the same folder as all of the .py files!


Our classifiers auto train on initializations.
To classify a dataset, just call the Classifier.classify() function.
Refer to hw5.py for examples on how to run each classifier.

Here are the parameters of each class:

DecisionTree(data, labels, validFeatures=ALL, impurityMeasure=infoImpurity, maxDepth=None, sampleFeatures=False, sampleData=False, prune=False)

RandomForst(X, y, M=100)

AdaBoost(data, y, Classifier=DecisionTree, params=[0,0,0,0])
	-Note that params was a hacky approach to passing parameters to Classifier. In the end that was scrapped, but we still used it as an indicator for the number of iterations. Ex: A length 4 array with all zeros (as shown above) will do 4 iterations.

The crossValidate function will automatically pass keyword argumented to the classifier.
Ex: crossValidate(X,y,DecisionTree,maxDepth=25) will pass the maxDepth parameter to DecisionTree.