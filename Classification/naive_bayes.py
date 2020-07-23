from model import Model
from sklearn.naive_bayes import GaussianNB


class NaiveBayes(Model):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.model = GaussianNB()
        self.name = "Naive Bayes"
