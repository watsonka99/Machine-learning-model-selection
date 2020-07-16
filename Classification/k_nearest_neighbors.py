from Classification.classification import Classification
from sklearn.neighbors import KNeighborsClassifier


class KNearestNeighbor(Classification):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)