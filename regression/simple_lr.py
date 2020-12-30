import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from scipy.stats import norm


class SimpleLinearModel(object):
    def __init__(self):
        self.X = None
        self.y = None
        self.xbar = None
        self.ybar = None
        self.b0 = None
        self.b1 = None

    def fit(self, features: np.array, target: np.array):
        """
        fit the linear model
        :param features: features
        :param target: target
        :return:
        """
        self.X = features
        self.y = target

        self.xbar = np.mean(self.X)
        self.ybar = np.mean(self.y)

        self._covariance()
        self._variance()

    def _covariance(self) -> None:
        """ calculate covariance """
        self.b1 = np.sum((self.X - self.xbar) * (self.y - self.ybar)) / np.sum(np.power(self.X - self.xbar, 2))

    def _variance(self) -> None:
        """ calculate variance """
        self.b0 = self.ybar - (self.b1 * self.xbar)

    def predict(self, features) -> np.array:
        """ predict regression line using exiting model """
        return self.b0 + self.b1 * features

    @staticmethod
    def _squared_error(y, yhat) -> np.array:
        """ calculate squared error """
        return sum((yhat - y)**2)

    def _r_squared(self, y, yhat) -> float:
        """ calculate coefficient of determination """
        y_mean = np.mean(y)
        y_line = [y_mean for _ in y]

        se_yhat = self._squared_error(y, yhat)
        se_y_mean = self._squared_error(y, y_line)

        return 1 - (se_yhat / se_y_mean)

    def plot(self, X, y, yhat) -> None:
        """ plot regression line """
        plt.style.use('ggplot')

        r2 = self._r_squared(y, yhat)
        conf = norm.interval(0.95, loc=np.mean(yhat), scale=yhat.std())

        plt.scatter(X, y, color='black')  # actual values
        plt.plot(X, yhat)  # regression line
        plt.fill_between(X.reshape(-1), (yhat+conf[0]), (yhat+conf[1]), color='b', alpha=0.2)

        # Labels
        plt.text(X.min().min(), y.max().max(), '$r^{2}$ = %s' % round(r2, 2))  # r squared
        plt.text(X.min().min(), y.max().max()-10, '95% confidence $\pm$ {:.2f}'.format(abs(conf[0])))  # r squared
        plt.title('Simple Linear Regression')
        plt.ylabel('Target (y)')
        plt.xlabel('Feature (X)')
        plt.show()


if __name__ == '__main__':
    n_samples = 1000
    train_size = int(n_samples * 0.80)

    # generate regression dataset
    X, y = make_regression(n_samples=1000, n_features=1, n_targets=1, random_state=42, noise=10)
    X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

    # Fit regression line
    model = SimpleLinearModel()
    model.fit(X_train.reshape(-1,), y_train)
    predictions = model.predict(X_test.reshape(-1,))
    model.plot(X_test, y_test, predictions)

