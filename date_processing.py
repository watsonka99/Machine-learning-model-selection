# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from Classification.kernel_svm import KernelSVM
from Classification.logistic_regression import LogisticRegressionClassification
from Classification.support_vector_machine import SupportVectorMachine
from Regression.decision_tree_regression import DecisionTree
from Regression.multiple_linear_regression import MultipleLinearRegression
from Regression.polynomial_regression import PolynomialRegression
from Regression.random_forest_regression import RandomForestRegression
from Regression.support_vector_regression import SupportVectorRegression
from sklearn.preprocessing import StandardScaler


class DataProcessing:

    def __init__(self):
        # Importing the dataset
        sc = StandardScaler()

        dataset = pd.read_csv('Data.csv')
        self.X = dataset.iloc[:, :-1].values
        self.y = dataset.iloc[:, -1].values

        # Splitting the dataset into the Training set and Test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=0)

        self.X_train_sc = sc.fit_transform(self.X_train)
        self.X_test_sc = sc.transform(self.X_test)

    def classification(self):
        kernel_SVM = LogisticRegressionClassification(self.X_train_sc, self.y_train, self.X_test_sc, self.y_test)
        kernel_SVM.train()
        print(kernel_SVM.accuracy_score())

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

        print("Regression Type                   || R2 Score")
        print("----------------------------------++--------------------")
        print("Multiple Linear Regression        ||", multiple_linear_regression.r2_score())
        print("Polynomial Regression             ||", polynomial_regression.r2_score())
        print("Support Vector Regression         ||", SVR.r2_score())
        print("Random Forest Regression          ||", random_tree.r2_score())
        print("Decision Tree Regression          ||", decesion_tree.r2_score())


if __name__ == '__main__':
    data = DataProcessing()
    data.classification()
