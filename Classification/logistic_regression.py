from Classification.classification import Classification
from sklearn.linear_model import LogisticRegression


class LogisticRegressionClassification(Classification):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.classifier = LogisticRegression(random_state=0)

