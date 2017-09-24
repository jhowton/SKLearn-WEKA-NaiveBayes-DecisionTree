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
n_data = pd.read_csv("nursery.data", header = None, names = ['parents', 
	'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 
	'health', 'class'])

#convert categorical features to binary
n_data_bin = pd.get_dummies(n_data.iloc[:, :-1]).values 

# Get classification for each row
n_data_classes = n_data.iloc[:, -1].values 

#Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(n_data_bin, 
	n_data_classes, test_size = 0.1)


#------------------- Naive Bayes ----------------------------------------------
#make NB object
gnb = GaussianNB() 

#fit model to training set
n_fit = gnb.fit(X_train, y_train) 

#score model using test set
nb_score = n_fit.score(X_test, y_test) 

#Print results
print("The Naive Bayes Score for nursery data set was {:.3f}".format(nb_score))



#------------------- Decision Tree --------------------------------------------
#make DT object
clf = tree.DecisionTreeClassifier()

#fit model to training set
n_fit_tree = clf.fit(X_train, y_train)

#Score model using test set
tree_score = n_fit_tree.score(X_test, y_test)

#Print results
print("The Decision Tree Score for nursery data set was {:.3f}".format(tree_score))

#Get binary feature names for writing tree to pdf
feature_names = list(pd.get_dummies(n_data.iloc[:,:-1]))

#Get class names for writing tree to pdf
class_names = n_fit_tree.classes_

#Generate dot data for making tree graphic
dot_data = tree.export_graphviz(n_fit_tree, 
	out_file = None, 
	feature_names = np.array(feature_names), #Show feature names
	class_names = class_names, #show class names
	filled=True, #Fill with colors based on class
	rounded = True, #Rounded nodes
	special_characters = True) #Allow for special characters

#Generate graph object from dot data
graph = pydotplus.graph_from_dot_data(dot_data)

#Write graph to pdf
graph.write_png("nursery_tree.png")