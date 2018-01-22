import pandas as pd
import LoanPrediction.Preprocessing as preproc
import numpy as np
from XGBooster import XGBooster
from LoanPrediction import SolveTest

train_data = pd.read_csv('../data/Loans/train.csv', sep=",", header='infer')
# Drop columns/Deal with NaN
train_data = preproc.process(train_data)

# Sklearn accepts only pure numerical data
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

model = XGBooster().work(features, target, n_estimators=100, learning_rate=0.05)

#preprocess same as training data
test_data = pd.read_csv('../data/Loans/test.csv', sep=",", header='infer')
data_ids = test_data['Loan_ID']
test_data = preproc.process(test_data)
test_data = preproc.label_encode(test_data, columns=['Dependents'])
test_data = pd.get_dummies(test_data, columns=["Gender", "Married", "Education",
                                               "Self_Employed", "Property_Area"])


SolveTest.solve(model, test_data, ids=data_ids, outfile="xgb_solutions.csv")