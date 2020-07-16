from sklearn.tree import DecisionTreeClassifier
from Classification.classification import Classification


class DecisionTreeClassification(Classification):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
