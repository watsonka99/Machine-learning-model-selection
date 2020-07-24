from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

class PolynomialRegression:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None
        self.degree = 0

    def train(self):
        best_degree_accuracy = 0
        self.degree = 1
        for x in range(1, 11):
            poly_reg = PolynomialFeatures(degree=x)
            X_poly = poly_reg.fit_transform(self.X_train)
            regressor = LinearRegression()
            regressor.fit(X_poly, self.y_train)
            self.y_pred = regressor.predict(poly_reg.transform(self.X_test))
            if best_degree_accuracy < accuracy_score(self.y_test, self.y_pred):
                self.degree = x
        #probarbly not optimal
        poly_reg = PolynomialFeatures(degree=x)
        X_poly = poly_reg.fit_transform(self.X_train)
        regressor = LinearRegression()
        regressor.fit(X_poly, self.y_train)
        self.y_pred = regressor.predict(poly_reg.transform(self.X_test))

