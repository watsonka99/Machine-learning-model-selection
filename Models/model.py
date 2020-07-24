from sklearn.metrics import confusion_matrix, accuracy_score, r2_score
from sklearn.model_selection import cross_val_score


class Model:

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None
        self.model = None
        self.accuracies = None
        self.accuracy = None
        self.sd = None
        self.name = None

    def train(self):
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.accuracies = cross_val_score(estimator=self.model, X=self.X_train, y=self.y_train, cv=10)
        self.accuracy = self.accuracies.mean() * 100
        self.sd = self.accuracies.std() * 100

    def r2_score(self):
        return r2_score(self.y_test, self.y_pred)

    def confusion_matrix(self):
        return confusion_matrix(self.y_test, self.y_pred)

    def accuracy_score(self):
        return accuracy_score(self.y_test, self.y_pred)
