from sklearn import preprocessing
import pandas as pd

def process(data):
    # Not relevant for prediction so dropped
    data = data.drop('Loan_ID', 1)

    # Handle NAN
    data = data.fillna({"Dependents": "0",
                                    "Married": "No",
                                    "Credit_History": round(data['Credit_History'].mean())})

    data = data.dropna(axis=0)
    return data

def label_encode(data, columns):
    le = preprocessing.LabelEncoder()
    for col in columns:
        le.fit(data[col].values)
        data[col] = le.transform(data[col].values)
    return data


def one_hot_encode(data, columns=[]):
    if columns == []:
        columns = data.columns.values
    data = pd.get_dummies(data, columns=columns)
    return data