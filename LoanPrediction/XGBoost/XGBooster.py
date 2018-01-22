import xgboost as xgb
from sklearn.model_selection import cross_val_score

#https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/
#https://www.kaggle.com/dansbecker/learning-to-use-xgboost
class XGBooster:

    @staticmethod
    def work(features, target, n_estimators, learning_rate):
        model = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=4)
        model.fit(features, target)
        print(model)
        XGBooster._report_metrics(model, features, target)
        return model

    # TODO make this inheritable for every mlearmner
    @staticmethod
    def _report_metrics(model, features, target):
        # perform cross validation
        scores = cross_val_score(model, features, target, cv=3)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        return scores.mean(), (scores.std() * 2)

