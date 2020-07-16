from sklearn.linear_model import LogisticRegression
from model import Model


class LogisticRegressionClassification(Model):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.model = LogisticRegression(random_state=0)

