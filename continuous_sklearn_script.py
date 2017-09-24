import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydotplus

#Not on MTSU Systems -----------------------------------------
# import pydotplus

#Read CSV
l_data = pd.read_csv("leaf.csv")

#Split into features and classes
l_features, l_classes = l_data.iloc[:, :-1], l_data.iloc[:, -1]

#Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(l_features, 
	l_classes, test_size = 0.1)


#------------------- Naive Bayes ----------------------------------------------
#make NB object
gnb = GaussianNB() 

#fit model to training set
l_fit_nb = gnb.fit(X_train, y_train)

#score model using test set
l_score = l_fit_nb.score(X_test, y_test)

#Print results
print("The Naive Bayes Score for leaf data set was {:.3f}".format(l_score))



#------------------- Decision Tree --------------------------------------------
#make DT object
clf = tree.DecisionTreeClassifier()

#fit model to training set
l_fit_tree = dt.fit(X_train, y_train)

#Score model using test set
tree_score = l_fit_tree.score(X_test, y_test)

#Print results
print("The Decision Tree Score for leaf data set was {:.3f}".format(tree_score))

#Get feature names for writing tree to pdf
feature_names = list(l_features)

#Get class names for writing tree to pdf
classes = [str(x) for x in list(l_fit_tree.classes_)]

#Generate dot data for making tree graphic
dot_data = tree.export_graphviz(l_fit_tree, 
	out_file = None, 
	feature_names = np.array(feature_names), #show feature names
	class_names = classes, #Show classes
	filled=True, #color by class
	rounded = True, #rounded nodes
	special_characters = True) #Allow special characters

#Generate graph object from dot data
graph = pydotplus.graph_from_dot_data(dot_data)

#Write graph to pdf
graph.write_png("leaf_tree.png")
