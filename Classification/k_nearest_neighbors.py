from sklearn.neighbors import KNeighborsClassifier
from model import Model


class KNearestNeighbor(Model):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)