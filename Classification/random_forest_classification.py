from sklearn.ensemble import RandomForestClassifier
from Classification.classification import Classification


class RandomForestClassification(Classification):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
