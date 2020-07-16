from sklearn.tree import DecisionTreeRegressor
from model import Model


class DecisionTree(Model):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.model = DecisionTreeRegressor(random_state=0)
