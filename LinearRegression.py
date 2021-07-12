# %%
from sklearn import datasets
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt

X, y = datasets.load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = model_selection.train_test_split(
    X_train, y_train, test_size=0.25
)  # 0.25 x 0.8 = 0.2


def run_model_on_data(model):
    # 	- fit a linear regression model on the training set
    params = model.fit(X_train, y_train)

    # 	- score your model on the training set
    training_score = model.score(X_train, y_train)
    #         print(f"training_score: {training_score}")

    # 	- score your model on the test set
    testing_score = model.score(X_test, y_test)
    #         print(f"testing_score: {testing_score}")

    training_score = model.score(X_train, y_train)
    #         print(f"training_score: {training_score}")

    validation_score = model.score(X_val, y_val)
    #         print(f"validation_score: {validation_score}")

    return training_score, testing_score, validation_score


class DataLoader:
    def __init__(self, X, y, batch_size=16):
        self.batches = []
        idx = 0
        while idx < len(X):
            self.batches.append(
                (X[idx : idx + batch_size], y[idx : idx + batch_size])
            )  # get batch
            idx += batch_size

    def __getitem__(self, idx):
        return self.batches[idx]  # X, y


class LinearRegressorHomeMade:
    def __init__(self, normalize=False, epochs=100, learning_rate=0.01) -> None:
        self.epochs = epochs
        self.normalize = normalize
        self.learning_rate = learning_rate

    def initialise_w_b(self, X):
        # y = wX + b
        print(f"Size of x:{X.shape}")
        self.w = np.random.randn(X.shape[1])
        self.b = np.random.randn()

    def fit(self, X, y):
        self.initialise_w_b(X)
        if self.normalize:
            X_mean = np.mean(X)
            X_std = np.std(X)
            X = (X - X_mean) / X_std

        batched_results = DataLoader(X, y, batch_size=100)

        losses = []
        # Epoch means we've made predictions of all the data in the dataset
        for epoch in range(self.epochs):
            for X, y in batched_results:
                # print(f"X.shape: {X.shape} -- y.shape: {y.shape}")
                y_hat = self.predict(X)
                loss = self.calculate_mse(y, y_hat)

                losses.append(loss)

                grad_w, grad_b = self.calculate_gradients(X, y)
                self.w -= grad_w * self.learning_rate
                self.b -= grad_b * self.learning_rate
            # print("---")

        print(f"Loss: {loss}")
        plt.plot(losses)
        plt.show()

    def calculate_mse(self, y, y_hat):
        mse = np.mean((y_hat - y) ** 2)
        return mse

    def score(self, X_test, y_actual):
        y_hat = self.predict(X_test)
        error = self.calculate_mse(y_actual, y_hat)

        return error

    def calculate_gradients(self, X, y):
        y_hat = self.predict(X)

        gradient_w = np.mean(2 * np.matmul((y_hat - y), X))  # not sure this is right
        gradient_b = 2 * np.mean(y_hat - y)

        return gradient_w, gradient_b

    def predict(self, X):
        # y = wX + b
        y_hat = np.matmul(X, self.w) + self.b
        # print(f"y_hat shape:{y_hat.shape}")
        return y_hat


def main():
    model = LinearRegressorHomeMade(normalize=True)
    training_score, testing_score, validation_score = run_model_on_data(model)

    print(
        f"training_score {training_score} \n testing_score {testing_score} \n validation_score {validation_score} \n"
    )


main()

# %%
