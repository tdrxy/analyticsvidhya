import pandas as pd

def explore_csv(filepath):
    data = pd.read_csv(filepath, sep=",", header='infer')

    print("Rows with NaN:" + str(data.shape[0] - data.dropna().shape[0]))
    print("What columns contain NaN?")
    print(data.isnull().any())
    print("\n")
    print("Dtypes:")
    print(data.dtypes)
    print("\n")

    print("## END EXPLORATION ##")


