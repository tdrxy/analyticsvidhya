# Reference:
# http://dataaspirant.com/2017/02/01/decision-tree-algorithm-python-with-scikit-learn/
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

class DecisionTree:
    @staticmethod
    def _write_tree(model, features):
        # Save tree in graphviz form
        dotfile = open("DT/tree.dot", 'w+')
        tree.export_graphviz(model, out_file=dotfile,
                             feature_names=features.columns.values)
        dotfile.close()

    @staticmethod
    def _report_metrics(model, features, target):
        # perform cross validation
        scores = cross_val_score(model, features, target, cv=7)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

class SimpleDecisionTree(DecisionTree):

    @staticmethod
    def work(features, target):
        dt_informationgain = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=3)
        dt_informationgain.fit(features, target)

        DecisionTree._report_metrics(dt_informationgain, features, target)
        DecisionTree._write_tree(dt_informationgain, features)
        return dt_informationgain










