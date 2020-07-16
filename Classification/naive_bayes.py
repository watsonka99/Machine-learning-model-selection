from Classification.classification import Classification
from sklearn.naive_bayes import GaussianNB


class NaiveBayes(Classification):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.classifier = GaussianNB()
