from sklearn.linear_model import LinearRegression

from model import Model


class MultipleLinearRegression(Model):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.model = LinearRegression()
