# Importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from Models.Both.ANN import ANN
from Models.Both.XGboost import XGboost
from Models.Classification.decision_tree_classification import DecisionTreeClassification
from Models.Classification.k_nearest_neighbors import KNearestNeighbor
from Models.Classification.kernel_svm import KernelSVM
from Models.Classification.logistic_regression import LogisticRegressionClassification
from Models.Classification.naive_bayes import NaiveBayes
from Models.Classification.random_forest_classification import RandomForestClassification
from Models.Classification.support_vector_machine import SupportVectorMachine
from Models.Regression.decision_tree_regression import DecisionTree
from Models.Regression.multiple_linear_regression import MultipleLinearRegression
from Models.Regression.polynomial_regression import PolynomialRegression
from Models.Regression.random_forest_regression import RandomForestRegression
from Models.Regression.support_vector_regression import SupportVectorRegression


class DataProcessing:

    def __init__(self):

        sc = StandardScaler()
        # Importing the dataset
        dataset = pd.read_csv('Data.csv')
        self.X = dataset.iloc[:, :-1].values
        self.y = dataset.iloc[:, -1].values

        # Splitting the dataset into the Training set and Test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=0)

        self.X_train_sc = sc.fit_transform(self.X_train)
        self.X_test_sc = sc.transform(self.X_test)
        self.model = []

    def classification(self):
        self.model.append(LogisticRegressionClassification(self.X_train_sc, self.y_train, self.X_test_sc,
                                                      self.y_test))
        self.model.append(KernelSVM(self.X_train_sc, self.y_train, self.X_test_sc, self.y_test))
        self.model.append(KNearestNeighbor(self.X_train_sc, self.y_train, self.X_test_sc, self.y_test))
        self.model.append(DecisionTreeClassification(self.X_train_sc, self.y_train, self.X_test_sc, self.y_test))
        self.model.append(NaiveBayes(self.X_train_sc, self.y_train, self.X_test_sc, self.y_test))
        self.model.append(RandomForestClassification(self.X_train_sc, self.y_train, self.X_test_sc, self.y_test))
        self.model.append(SupportVectorMachine(self.X_train_sc, self.y_train, self.X_test_sc, self.y_test))
        self.model.append(XGboost(self.X_train, self.y_train, self.X_test, self.y_test))
  #      self.model.append(ANN(self.X_train_sc, self.y_train, self.X_test_sc, self.y_test))

    def regression(self):
        self.model.append(DecisionTree(self.X_train, self.y_train, self.X_test, self.y_test))
        self.model.append(MultipleLinearRegression(self.X_train, self.y_train, self.X_test, self.y_test))
        self.model.append(RandomForestRegression(self.X_train, self.y_train, self.X_test, self.y_test))
        self.model.append(SupportVectorRegression(self.X, self.y))
        self.model.append(PolynomialRegression(self.X_train, self.y_train, self.X_test, self.y_test))
        self.model.append(XGboost(self.X_train, self.y_train, self.X_test, self.y_test))

    def train(self):
        for x in self.model:
            x.train()

    def graph(self):
        plt.boxplot([x.accuracies * 100 for x in self.model], patch_artist=True,
                    labels=[x.name for x in self.model])
        plt.title("Model Comparison")
        plt.ylabel("Accuracy (%)")
        plt.ylim(60, 100)
        plt.xticks(rotation=90)
        plt.show()


if __name__ == '__main__':
    data = DataProcessing()
    data.classification()
    data.train()
  #  data.graph()
