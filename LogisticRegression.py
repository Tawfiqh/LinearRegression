# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from LinearRegression import LinearRegressorHomeMade, DataLoader
from sklearn.datasets import load_breast_cancer


class LogisticRegressorHomeMade(LinearRegressorHomeMade):
    def _init_(self) -> None:
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
        epochs = 50
        self.initialise_w_b(X)

        losses = []
        # Epoch means we've made predictions of all the data in the dataset
        for epoch in range(epochs):
            loss = self._calculate_bce_loss_(X, y)
            print(f"Loss:{loss}")

            grad_w, grad_b = self.calculate_gradients(X, y)
            print(f"grad_w - shape: {grad_w}")
            print(f"grad_b - shape: {grad_b}")
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
    def _predict_linear_(self, X):
        # y = wX + b
        # print(f"self.w shape:{self.w.shape}")
        # print(f"X shape:{X.shape}")
        # print(f"self.b:{self.b}")
        y_hat = np.matmul(X, self.w) + self.b
        # print(f"y_hat shape:{y_hat.shape}")
        return y_hat

    def _sigmoid_(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        return self._sigmoid_(self._predict_linear_(X))

    # This should be different
    def _calculate_bce_loss_(self, X, y):
        y_hat = self.predict(X)
        loss = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return np.mean(loss)

        x_whole = self._predict_linear_(X)
        print(f"x = self._predict_linear_(X):\n {x}\n")
        for x in x_whole:
            loss = max(0, x) - (x * y) + (np.log(1 + np.exp(-1 * abs(x))))
        return loss

    # Score would be different
    def score(self, X_test, y_actual):
        return 0

    def _dl_dy_hat_(self, y, y_hat):
        return -1 * ((y * (1 / y_hat)) - ((1 - y) * (1 / (1 - y_hat))))

    def _dy_hat_dz_(self, z):
        sigma_z = self._sigmoid_(z)
        return sigma_z * (1 - sigma_z)

    # This should be different for logistic regressor
    def calculate_gradients(self, X, y):
        y_hat = self.predict(X)
        z = self._predict_linear_(X)  # Old y_hat i.e old prediction = Xw+b

        dl_dy_hat = self._dl_dy_hat_(y, y_hat)
        dy_hat_dz = self._dy_hat_dz_(z)
        dz_dw = X
        dz_db = 1

        gradient_w = np.matmul(dl_dy_hat, dy_hat_dz) * dz_dw
        gradient_w = np.mean(gradient_w)
        gradient_b = np.matmul(dl_dy_hat, dy_hat_dz) * dz_db

        return gradient_w, gradient_b


def main():
    # X, y = load_iris(return_X_y=True)
    X, y = load_breast_cancer(return_X_y=True)

    normalize = True
    if normalize:
        X_mean = np.mean(X)
        X_std = np.std(X)
        X = (X - X_mean) / X_std

    # print(f"x:{X}")
    # print(f"y:{y}")

    #   - load it in, split it into train, test and val
    #   - take a look at an example datapoint’s features and labels

    model = LogisticRegressorHomeMade()
    model.fit(X, y)

    # predict = model.predict(X[:2, :])
    # print(f"predict result: {predict}\n\n")

    # predict_proba = model.predict_proba(X[:2, :])
    # print(f"predict_proba result: {predict_proba}\n\n")

    score = model.score(X, y)
    print(f"score result: {score}\n\n")


main()

