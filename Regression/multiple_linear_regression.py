from sklearn.linear_model import LinearRegression

from Regression.regression import Regression


class MultipleLinearRegression(Regression):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.regressor = LinearRegression()
