from sklearn.linear_model import LogisticRegression
from Models.model import Model


class LogisticRegressionClassification(Model):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.model = LogisticRegression()
        self.name = "Logistic Regression"

