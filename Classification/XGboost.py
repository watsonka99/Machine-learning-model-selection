from xgboost import XGBClassifier

from model import Model


class XGboost(Model):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.model = XGBClassifier()
        self.name = "XG boost"
