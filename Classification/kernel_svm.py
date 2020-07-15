from Classification.classification import Classification
from sklearn.svm import SVC


class KernelSVM(Classification):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.classifier = SVC(kernel = 'rbf', random_state = 0)
