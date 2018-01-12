from sklearn import preprocessing

def process(data):
    # Not relevant for prediction so dropped
    train_data = data.drop('Loan_ID', 1)

    # Handle NAN
    train_data = train_data.fillna({"Dependents": "0",
                                    "Married": "No",
                                    "Credit_History": round(train_data['Credit_History'].mean())})

    train_data = train_data.dropna(axis=0)
    return train_data