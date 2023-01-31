import numpy as np


class LinearRegression:
    """
    Linear Regression model
    Has 2 parameters: w and b
    """

    w: np.ndarray
    b: float

    def __init__(self):
        """Empty init function -> no random initialization needed"""
        self.w = np.random.randn(
            1,
        )
        self.b = np.random.randn(
            1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the data to the closed form solution for
        linear regression. Sets the values for w and b
        """
        ones = np.ones((X.shape[0], 1))
        transformed_X = np.concatenate((ones, X), axis=1)
        # print(transformed_X.shape)
        wb = np.linalg.inv(transformed_X.T @ transformed_X) @ transformed_X.T @ y

        self.w = wb[1:]
        self.b = float(wb[0])
        # print(self.w)
        # print(self.b)
        return self.w, self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """predicts the y values given the input
        using the parameters fit in the fit()
        function
        """
        return (X @ self.w) + self.b
        # print((X @ self.w) + self.b)
        # print(preds.shape)
        # print(X.shape)
        # raise NotImplementedError()


class GradientDescentLinearRegression(LinearRegression):
    """
    Gradient Descent Linear Regression model
    Has 4 parameters: w, b, lr, and epochs
    """

    w: np.ndarray
    b: float

    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        Fits the data to the closed form solution for
        linear regression using gradient descent. Sets the values for w and b
        """
        self.w = np.random.randn(
            X.shape[1],
        )
        self.b = np.random.randn(
            1,
        )
        N = X.shape[0]
        for e in range(epochs):
            # print('y true: ', y.shape)
            preds = np.squeeze((X @ self.w) + self.b)
            # print('preds: ', preds.shape)
            # print('X.T: ', X.T.shape)
            # print('(y - preds): ', (y - preds).shape)
            delta_w = np.clip((-2 / N) * (X.T @ (y - preds)), -1, 1)
            delta_b = (-2 / N) * np.sum((y - preds), axis=0)
            self.w = self.w - (lr * delta_w)
            self.b = self.b - (lr * delta_b)
            # print('w: ', self.w.shape)
            # print('b: ', self.b.shape)
        print("w: ", self.w)
        print("b: ", self.b)
        return self.w, self.b
        # raise NotImplementedError()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return (X @ self.w) + self.b
        # raise NotImplementedError()
