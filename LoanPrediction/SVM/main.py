import pandas as pd
import LoanPrediction.Preprocessing as preproc
import SVMExperiments
import numpy as np

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

#target = np.argmax(target, axis=1)

SVMExperiments.test_parameters(features, target, kernel = "linear", Cs=[1, 50, 100], gammas=[1,5,15])
