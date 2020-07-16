from xgboost import XGBClassifier

from Classification.classification import Classification


class XGboost(Classification):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.classifier = XGBClassifier()
