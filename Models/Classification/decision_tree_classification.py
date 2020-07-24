from sklearn.tree import DecisionTreeClassifier
from Models.model import Model


class DecisionTreeClassification(Model):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.model = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
        self.name = "Decision Tree"
