from sklearn.ensemble import RandomForestRegressor

from Regression.regression import Regression


class RandomForestRegression(Regression):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

