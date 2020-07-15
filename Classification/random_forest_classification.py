from sklearn.ensemble import RandomForestClassifier
from Classification.classification import Classification


class RandomForestClassification(Classification):

    def train(self):
        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        classifier.fit(self.X_train, self.y_train)
        self.y_pred = classifier.predict(self.X_test)
