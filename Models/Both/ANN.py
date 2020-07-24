import tensorflow as tf
from sklearn.model_selection import cross_val_score

from Models.model import Model


class ANN(Model):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.model = tf.keras.models.Sequential()
        # Adding the input layer and the first hidden layer
        self.model.add(tf.keras.layers.Dense(units=6, activation='relu'))
        # Adding the second hidden layer
        self.model.add(tf.keras.layers.Dense(units=6, activation='relu'))
        # Adding the output layer
        self.model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        self.name = "ANN"

    def train(self):
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(self.X_train, self.y_train, batch_size = 32, epochs = 100)
        self.y_pred = self.model.predict(self.X_test)
        self.accuracies = cross_val_score(estimator=self.model, X=self.X_train, y=self.y_train, cv=10)
      #  self.accuracy = self.accuracies.mean() * 100
      #  self.sd = self.accuracies.std() * 100