# Importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from Classification.XGboost import XGboost
from Classification.decision_tree_classification import DecisionTreeClassification
from Classification.k_nearest_neighbors import KNearestNeighbor
from Classification.kernel_svm import KernelSVM
from Classification.logistic_regression import LogisticRegressionClassification
from Classification.naive_bayes import NaiveBayes
from Classification.random_forest_classification import RandomForestClassification
from Classification.support_vector_machine import SupportVectorMachine
from Regression.decision_tree_regression import DecisionTree
from Regression.multiple_linear_regression import MultipleLinearRegression
from Regression.polynomial_regression import PolynomialRegression
from Regression.random_forest_regression import RandomForestRegression
from Regression.support_vector_regression import SupportVectorRegression


class DataProcessing:

    def __init__(self):
        # Importing the dataset
        sc = StandardScaler()

        dataset = pd.read_csv('pd_speech_features.csv')
        self.X = dataset.iloc[:, :-1].values
        self.y = dataset.iloc[:, -1].values

        # Splitting the dataset into the Training set and Test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=0)

        self.X_train_sc = sc.fit_transform(self.X_train)
        self.X_test_sc = sc.transform(self.X_test)

    def classification(self):
        model = []
        model.append(LogisticRegressionClassification(self.X_train_sc, self.y_train, self.X_test_sc,
                                                               self.y_test))
        model.append(KernelSVM(self.X_train_sc, self.y_train, self.X_test_sc, self.y_test))
        model.append(KNearestNeighbor(self.X_train_sc, self.y_train, self.X_test_sc, self.y_test))
        model.append(DecisionTreeClassification(self.X_train_sc, self.y_train, self.X_test_sc, self.y_test))
        model.append(NaiveBayes(self.X_train_sc, self.y_train, self.X_test_sc, self.y_test))
        model.append(RandomForestClassification(self.X_train_sc, self.y_train, self.X_test_sc, self.y_test))
        model.append(SupportVectorMachine(self.X_train_sc, self.y_train, self.X_test_sc, self.y_test))
        model.append(XGboost(self.X_train, self.y_train, self.X_test, self.y_test))
        return model

    def regression(self):
        decesion_tree = DecisionTree(self.X_train, self.y_train, self.X_test, self.y_test)
        decesion_tree.train()

        multiple_linear_regression = MultipleLinearRegression(self.X_train, self.y_train, self.X_test, self.y_test)
        multiple_linear_regression.train()

        random_tree = RandomForestRegression(self.X_train, self.y_train, self.X_test, self.y_test)
        random_tree.train()

        SVR = SupportVectorRegression(self.X, self.y)
        SVR.train()

        polynomial_regression = PolynomialRegression(self.X_train, self.y_train, self.X_test, self.y_test)
        polynomial_regression.train()

    def train(self, models):
        for model in models:
            model.train()

    def graph(self, box_plot_data):
        plt.boxplot([model.accuracies for model in box_plot_data], patch_artist=True, labels=[model.name for model in box_plot_data])
        plt.ylim(0.5, 1)
        plt.xticks(rotation=90)
        plt.show()





if __name__ == '__main__':
    data = DataProcessing()
    models = data.classification()
    data.train(models)
    data.graph(models)