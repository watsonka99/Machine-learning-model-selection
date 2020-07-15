from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score


class SupportVectorRegression:

    def __init__(self, X, y):
        self.sc_X = StandardScaler()
        self.sc_y = StandardScaler()
        self.X = X
        self.y = y.reshape(len(y),1)
        X_train, self.X_test, y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 0)
        self.X_train = self.sc_X.fit_transform(X_train)
        self.y_train = self.sc_y.fit_transform(y_train)

    def train(self):
        regressor = SVR(kernel = 'rbf')
        regressor.fit(self.X_train, self.y_train)
        self.y_pred = self.sc_y.inverse_transform(regressor.predict(self.sc_X.transform(self.X_test)))

    def r2_score(self):
        return r2_score(self.y_test, self.y_pred)


