from sklearn.tree import DecisionTreeClassifier
from Classification.classification import Classification


class DecisionTreeClassification(Classification):

    def train(self):
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(self.X_train, self.y_train)
        self.y_pred = classifier.predict(self.X_test)