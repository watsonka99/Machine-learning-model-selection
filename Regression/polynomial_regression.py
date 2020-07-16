from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class PolynomialRegression:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None

    def train(self):
        poly_reg = PolynomialFeatures(degree=4)
        X_poly = poly_reg.fit_transform(self.X_train)
        regressor = LinearRegression()
        regressor.fit(X_poly, self.y_train)
        self.y_pred = regressor.predict(poly_reg.transform(self.X_test))

