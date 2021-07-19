# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from LinearRegression import LinearRegressorHomeMade, DataLoader
from sklearn.datasets import load_breast_cancer
import math


class LogisticRegressorHomeMade(LinearRegressorHomeMade):
    def __init__(self) -> None:
        pass
        # self.lambda_rf = 0.3
        # TODO: lambda_rf
        # todo - normalisation

    def initialise_w_b(self, X):
        # y = wX + b
        # print(f"Size of x:{X.shape}")
        self.w = np.random.randn(X.shape[1])
        self.b = np.random.randn()

    # run the hyper-paramater tuning in a separate class -- paramaters are for the class not for the fit-function
    def fit(self, X, y, validation_set=None):
        learning_rate = 0.01
        epochs = 500
        self.initialise_w_b(X)

        losses = []
        # Epoch means we've made predictions of all the data in the dataset
        for epoch in range(epochs):
            loss = self.calculate_bce_loss(X, y)
            print(f"Loss:{loss}")

            grad_w, grad_b = self.calculate_gradients(X, y)
            self.w -= grad_w * learning_rate
            self.b -= grad_b * learning_rate

            # TODO: Early stopping
            losses.append(loss)
            # TODO: Validation-set + validation losses

        print(f"Loss: {loss}")
        plt.plot(losses)
        plt.legend(["Loss", "Validation losses"])
        plt.show()
        return loss

    # This should be different for logistic regressor
    def predict_linear(self, X):
        # y = wX + b
        # print(f"self.w shape:{self.w.shape}")
        # print(f"X shape:{X.shape}")
        # print(f"self.b:{self.b}")
        y_hat = np.matmul(X, self.w) + self.b
        # print(f"y_hat shape:{y_hat.shape}")
        return y_hat

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def predict(self, X):
        return self.sigmoid(self.predict_linear(X))

    # This should be different
    def calculate_bce_loss(self, X, y):
        y_hat = self.predict(X)
        loss = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return loss

    # Score would be different
    def score(self, X_test, y_actual):
        return 0

    def dl_dy_hat(self, y, y_hat):
        return -1 * ((y * (1 / y_hat)) - ((1 - y) * (1 / (1 - y_hat))))

    def dy_hat_dz(self, z):
        sigma_z = self.sigmoid(z)
        return sigma_z * (1 - sigma_z)

    # This should be different for logistic regressor
    def calculate_gradients(self, X, y):
        y_hat = self.predict(X)
        # print(f"X shape:{X.shape}")
        # print(f"y_hat shape:{y_hat.shape}")
        # print(f"y shape:{y.shape}")
        # temp = np.matmul((y_hat - y), X)
        # print(f"temp shape:{temp.shape}")
        z = self.predict_linear(X)  # Old y_hat i.e old prediction = Xw+b
        dl_dy_hat = self.dl_dy_hat(y, y_hat)
        dy_hat_dz = self.dy_hat_dz(z)
        dz_dw = X
        dz_db = 1

        gradient_w = dl_dy_hat * dy_hat_dz * dz_dw
        gradient_b = dl_dy_hat * dy_hat_dz * dz_db

        return gradient_w, gradient_b


def main():
    # X, y = load_iris(return_X_y=True)
    X, y = load_breast_cancer(return_X_y=True)
    # print(f"x:{X}")
    # print(f"y:{y}")
    model = LogisticRegressorHomeMade()
    model.fit(X, y)

    # predict = model.predict(X[:2, :])
    # print(f"predict result: {predict}\n\n")

    # predict_proba = model.predict_proba(X[:2, :])
    # print(f"predict_proba result: {predict_proba}\n\n")

    score = model.score(X, y)
    print(f"score result: {score}\n\n")


# Predictor is the Sigmoid OF wx+b

main()

