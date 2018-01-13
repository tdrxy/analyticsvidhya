from LoanPrediction import SolveTest
import pandas as pd
from RandomForest import RandomForest
from LoanPrediction.DT.DecisionTree import SimpleDecisionTree

import LoanPrediction.DT.Preprocessing as preproc
import SimpleDataExploration

# Explore data, printing some information
SimpleDataExploration.explore_csv('../data/Loans/train.csv')

train_data = pd.read_csv('../data/Loans/train.csv', sep=",", header='infer')
# Drop columns/Deal with NaN
train_data = preproc.process(train_data)

# Sklearn accepts only pure numerical data when dealing with Decision Trees
# Before one hot encoding, label encode the Dependent column, as this is ordinal
# (therefore sklean can see this as continuous)
preproc.label_encode(train_data, columns=['Dependents'])

target = train_data['Loan_Status']
features = train_data.drop('Loan_Status', 1)

# SKLEARN DOES NOT WORK WITH CATEGORICAL DATA
# One hot encoding is used!
# One hot encode categorical data
features = pd.get_dummies(features, columns=["Gender", "Married", "Education",
                                             "Self_Employed", "Property_Area"])
# Target
#  N Y
# 0 1 0
# 1 1 0
target = pd.get_dummies(target)

print("Simple Decision Tree: ")
simple_dt = SimpleDecisionTree().work(features, target)

print("Random Forest: ")
forest = RandomForest().work(features, target)

#preprocess same as training data
test_data = pd.read_csv('../data/Loans/test.csv', sep=",", header='infer')
data_ids = test_data['Loan_ID']
test_data = preproc.process(test_data)
test_data = preproc.label_encode(test_data, columns=['Dependents'])
test_data = pd.get_dummies(test_data, columns=["Gender", "Married", "Education",
                                               "Self_Employed", "Property_Area"])


SolveTest.solve(simple_dt, test_data, ids=data_ids, outfile="simple_solutions.csv")
SolveTest.solve(simple_dt, test_data, ids=data_ids, outfile="rf_solutions.csv")
