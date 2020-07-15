from Classification.classification import Classification
from sklearn.naive_bayes import GaussianNB


class NaiveBayes(Classification):

    def train(self):
        classifier = GaussianNB()
        classifier.fit(self.X_train, self.y_train)
        self.y_pred = classifier.predict(self.X_test)