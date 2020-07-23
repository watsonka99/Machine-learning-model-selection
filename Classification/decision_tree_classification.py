from sklearn.tree import DecisionTreeClassifier
from model import Model


class DecisionTreeClassification(Model):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.model = DecisionTreeClassifier(criterion = 'entropy')
        self.name = "Decision Tree"
