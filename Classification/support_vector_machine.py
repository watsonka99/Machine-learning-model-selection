from model import Model
from sklearn.svm import SVC


class SupportVectorMachine(Model):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.model = SVC(kernel = 'linear')
        self.name = "SVM"

