from sklearn.metrics import r2_score

from model import Model


class Regression(Model):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.regressor = None

    def train(self):
        self.regressor.fit(self.X_train, self.y_train)
        self.y_pred = self.regressor.predict(self.X_test)

    def r2_score(self):
        return r2_score(self.y_test, self.y_pred)
