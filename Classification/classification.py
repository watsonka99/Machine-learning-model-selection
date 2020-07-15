from sklearn.metrics import confusion_matrix, accuracy_score

class Classification:

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None
        self.classifier = None

    def train(self):
        self.classifier.fit(self.X_train, self.y_train)
        self.y_pred = self.classifier.predict(self.X_test)

    def confusion_matrix(self):
        return confusion_matrix(self.y_test, self.y_pred)

    def accuracy_score(self):
        return accuracy_score(self.y_test, self.y_pred)