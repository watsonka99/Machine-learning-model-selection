from sklearn.ensemble import RandomForestClassifier
from Models.model import Model


class RandomForestClassification(Model):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.model = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state=0)
        self.name = "Random Forest"
