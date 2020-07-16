from sklearn.metrics import confusion_matrix, accuracy_score


class Model:

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None

    def confusion_matrix(self):
        return confusion_matrix(self.y_test, self.y_pred)

    def accuracy_score(self):
        return accuracy_score(self.y_test, self.y_pred)


      #  from sklearn.model_selection import cross_val_score
       # accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
        #print("Accuracy: {:.2f} %".format(accuracies.mean() * 100))
        #print("Standard Deviation: {:.2f} %".format(accuracies.std() * 100))