import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from plot_functions import plot_regression


if __name__ == '__main__':

    # generate regression dataset
    n_samples = 1000
    X, y = make_regression(n_samples=1000, n_features=1, n_targets=1, random_state=42, noise=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, shuffle=False)

    # Build and train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    r2 = model.score(X_test, y_test)
    plot_regression(X_test, y_test, predictions, r2, 'Simple Linear Regression - sklearn')
