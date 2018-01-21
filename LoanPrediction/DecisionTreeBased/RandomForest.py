from sklearn.ensemble import RandomForestClassifier
from LoanPrediction.DecisionTreeBased.DecisionTree import DecisionTree

class RandomForest(DecisionTree):

    @staticmethod
    def work(features, target):
        model = RandomForestClassifier(n_estimators=1000)
        model.fit(features, target)

        DecisionTree._report_metrics(model, features, target)
        return model

