from sklearn.tree import DecisionTreeRegressor
from Regression.regression import Regression


class DecisionTree(Regression):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.regressor = DecisionTreeRegressor(random_state=0)
