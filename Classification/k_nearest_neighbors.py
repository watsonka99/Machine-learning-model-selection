from Classification.classification import Classification
from sklearn.neighbors import KNeighborsClassifier


class KNearestNeighbor(Classification):

    def train(self):
        classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        classifier.fit(self.X_train, self.y_train)
        self.y_pred = classifier.predict(self.X_test)