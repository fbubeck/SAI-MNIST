from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np
from matplotlib import pyplot as plt
from time import time
from sklearn.metrics import mean_squared_error

class PolyRegression():
    def __init__(self, train_data, test_data, lr_degree):
        self.train_data = train_data
        self.test_data = test_data
        self.varianz = self.test_data[2]
        self.train_error = np.empty(10)
        self.test_error = np.empty(10)
        self.degree = lr_degree

    def train(self):
        # Training Data
        self.xs_train = np.matrix(self.train_data[0]).T.A
        self.ys_train = np.matrix(self.train_data[1]).T.A


        # Modelfitting
        self.model = LinearRegression()
        self.poly = PolynomialFeatures(self.degree)
        xs_TRANS = self.poly.fit_transform(self.xs_train)
        start_training = time()
        #self.model.fit(self.xs_train, self.ys_train)
        self.model.fit(xs_TRANS, self.ys_train)
        end_training = time()

        # Time
        duration_training = end_training - start_training

        # Prediction for Training mse
        y_pred = self.model.predict(xs_TRANS)

        error = (mean_squared_error(self.ys_train, y_pred)/self.varianz)*100

        # Summary
        print('------ LinearRegression ------')
        print(f'Duration Training: {duration_training} seconds')
        print('Coefficients: ', self.model.coef_)
        print("Number of Parameter: ", self.degree + 1)

        return duration_training, error

    def test(self):
        # Test Data
        self.xs_test = np.matrix(self.test_data[0]).T.A
        self.ys_test = np.matrix(self.test_data[1]).T.A
        xs_TRANS = self.poly.fit_transform(self.xs_train)

        # Predictions
        start_test = time()
        y_pred = self.model.predict(xs_TRANS)
        end_test = time()

        # Time
        duration_test = end_test - start_test

        print(f'Duration Inference: {duration_test} seconds')

        # Error
        error = (mean_squared_error(self.ys_test, y_pred)/self.varianz)*100
        print("Mean squared error: %.2f" % error)
        print("")

        return duration_test, error, y_pred