from Classification.classification import Classification
from sklearn.svm import SVC


class SupportVectorMachine(Classification):

    def train(self):
        classifier = SVC(kernel = 'linear', random_state = 0)
        classifier.fit(self.X_train, self.y_train)
        self.y_pred = classifier.predict(self.X_test)

