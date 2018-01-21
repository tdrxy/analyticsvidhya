from sklearn import svm
from sklearn.model_selection import cross_val_score

class SVM:

    def work(self, features, target, kernel, C, gamma):
        # About dim of target:
        # https://stackoverflow.com/questions/45768899/sklearn-bad-input-shape-valueerror

        svc = svm.SVC(kernel=kernel, C=C, gamma=gamma)
        svc.fit(features, target)
        print("SVM trained")
        acc, std = self._report_metrics(svc, features, target)
        return svc, acc, std

    # TODO make this inheritable for every mlearmner
    @staticmethod
    def _report_metrics(model, features, target):
        # perform cross validation
        scores = cross_val_score(model, features, target, cv=3)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        return scores.mean(), (scores.std() * 2)

