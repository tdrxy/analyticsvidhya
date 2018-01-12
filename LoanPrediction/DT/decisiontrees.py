# Reference:
# http://dataaspirant.com/2017/02/01/decision-tree-algorithm-python-with-scikit-learn/
import pandas as pd
from sklearn import preprocessing
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

import LoanPrediction.DT.Preprocessing as preproc

# Explore data, printing some information
# SimpleDataExploration.explore_csv('../data/Loans/train.csv')

train_data = pd.read_csv('../data/Loans/train.csv', sep=",", header='infer')
# Drop columns/Deal with NaN
train_data = preproc.process(train_data)

# Sklearn accepts only pure numerical data when dealing with Decision Trees
# Before one hot encoding, label encode the Dependent column, as this is ordinal
# (therefore sklean can see this as continuous)
le = preprocessing.LabelEncoder()
le.fit(train_data['Dependents'].values)
train_data['Dependents'] = le.transform(train_data['Dependents'].values)

target = train_data['Loan_Status']
features = train_data.drop('Loan_Status', 1)

# SKLEARN DOES NOT WORK WITH CATEGORICAL DATA
# One hot encoding is used!
# One hot encode categorical data
features = pd.get_dummies(features, columns=["Gender", "Married", "Education",
                                   "Self_Employed", "Property_Area"])
target = pd.get_dummies(target)

dt_informationgain = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=3)
dt_informationgain.fit(features, target)

# perform cross validation
scores = cross_val_score(dt_informationgain, features, target, cv=7)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Safe tree in graphviz form
dotfile = open("tree.dot", 'w+')
tree.export_graphviz(dt_informationgain, out_file = dotfile, feature_names = features.columns.values)
dotfile.close()

