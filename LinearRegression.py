# %%
from sklearn import datasets
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X, y = datasets.load_boston(return_X_y=True)
print(f"Original X Size:{X.shape}")
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = model_selection.train_test_split(
    X_train, y_train, test_size=0.25
)  # 0.25 x 0.8 = 0.2


def run_model_on_data(model):
    # 	- fit a linear regression model on the training set
    print("TRAINING:")
    model.fit(X_train, y_train, validation_set=(X_test, y_test))

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
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle()

    # Todo -- call this automatically in the __getitem__
    def shuffle(self):
        batches = []
        idx = 0

        while idx < len(self.X):
            batches.append(
                (
                    self.X[idx : idx + self.batch_size],
                    self.y[idx : idx + self.batch_size],
                )
            )  # get batch
            idx += self.batch_size

        np.random.shuffle(batches)
        self.batches = batches

    def __getitem__(self, idx):
        return self.batches[idx]  # X, y


class LinearRegressorHomeMade:
    def __init__(self, normalize=False) -> None:
        self.normalize = normalize

    def initialise_w_b(self, X):
        # y = wX + b
        print(f"Size of x:{X.shape}")
        self.w = np.random.randn(X.shape[1])
        self.b = np.random.randn()

    def fit(self, X, y, validation_set=None):
        grid_search_results = []
        for learning_rate in range(0, 100, 5):
            learning_rate = learning_rate / 100
            for epochs in range(0, 1000, 50):
                loss = self._fit_grid_search_(
                    X,
                    y,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    validation_set=validation_set,
                )
                grid_search_results.append([loss, learning_rate, epochs])

        df = pd.DataFrame(
            grid_search_results, columns=["loss", "learning_rate", "epochs"]
        )
        pd.options.display.float_format = "{:,.4f}".format

        display(df)

    def _fit_grid_search_(
        self, X, y, learning_rate=0.01, epochs=5000, validation_set=None
    ):
        self.initialise_w_b(X)
        if self.normalize:
            X_mean = np.mean(X)
            X_std = np.std(X)
            X = (X - X_mean) / X_std
            if validation_set:
                X_val = validation_set[0]
                X_val = (X_val - X_mean) / X_std
                validation_set = (X_val, validation_set[1])

        batched_results = DataLoader(X, y, batch_size=100)

        losses = []
        validation_losses = []
        previous_loss = 0.0
        loss = 0.0
        # Epoch means we've made predictions of all the data in the dataset
        for epoch in range(epochs):
            batched_results.shuffle()
            for X, y in batched_results:
                # print(f"X.shape: {X.shape} -- y.shape: {y.shape}")
                loss = self.calculate_loss(X, y)
                # print(f"Loss:{loss}")

                grad_w, grad_b = self.calculate_gradients(X, y)
                self.w -= grad_w * learning_rate
                self.b -= grad_b * learning_rate
            # Early stopping
            if (previous_loss != 0) and (
                abs((previous_loss - loss) / previous_loss) < 0.0001
            ):
                percent = (previous_loss - loss) / previous_loss
                # print(f"Previous_loss: {previous_loss}")
                # print(f"Loss: {loss}")
                # print(f"Percent change: {percent}")
                # print(f"EARLY STOPPING! on epoch {epoch}")
                break
            previous_loss = loss
            losses.append(loss)
            if validation_set:
                validation_losses.append(self.calculate_loss(*validation_set))
            # print("---")

        print(f"Loss: {loss}")
        plt.plot(losses)
        plt.plot(validation_losses)
        plt.legend(["Loss", "Validation losses"])
        plt.show()
        return loss

    def calculate_mse(self, y, y_hat):
        mse = np.mean((y_hat - y) ** 2)
        return mse

    def calculate_var_y(self, y_actual):
        y_mean = np.mean(y_actual)
        return np.mean(y_actual - y_mean)

    def calculate_loss(self, X, y):
        y_hat = self.predict(X)
        loss = self.calculate_mse(y, y_hat)
        return loss

    def score(self, X_test, y_actual):
        y_hat = self.predict(X_test)
        mse = self.calculate_mse(y_actual, y_hat)
        var_y = self.calculate_var_y(y_actual)
        r_squared = 1 - (mse / var_y)

        return r_squared

    def calculate_gradients(self, X, y):
        y_hat = self.predict(X)

        gradient_w = np.mean(2 * np.matmul((y_hat - y), X))  # not sure this is right
        gradient_b = 2 * np.mean(y_hat - y)

        return gradient_w, gradient_b

    def predict(self, X):
        # y = wX + b
        # print(f"self.w shape:{self.w.shape}")
        # print(f"X shape:{X.shape}")
        # print(f"self.b:{self.b}")
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


# - from scratch bias-variance trade-off
#   - update your from scratch linear regression code to use a validation set
# Every iterations - calculate the loss on the training data
# Every epoch - calculate the loss on the validation data
# Plot those together.

#   - graph both the losses together on the same plot
#   - do they look as expected?
# NO!!! - validation loss is so high


#   - can you identify if your model is underfitting or overfitting to the data from these graphs?
# Pretty sure it's overfitting to the original training data

#   - which of these possibilities mean your model is biased or has high variance?
# High variance as it has overfit???

#   - add early stopping to your linear regression from scratch code
# Okay

#  implement a grid search
#   - update your from scratch linear regression code to include a grid search
# Grid-search -- over learning rates and batch size
#   - print the best hyperparameterisations
#   - initialise a model with them and train it
#   - save it, yes, your custom model. Does it work in the same way?

# %%
