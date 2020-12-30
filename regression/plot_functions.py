import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def plot_regression(X, y, yhat, r2, title):
    """ plot regression line """
    plt.style.use('ggplot')

    conf = norm.interval(0.95, loc=np.mean(yhat), scale=yhat.std())

    plt.scatter(X, y, color='black')  # actual values
    plt.plot(X, yhat)  # regression line
    plt.fill_between(X.reshape(-1), (yhat+conf[0]), (yhat+conf[1]), color='b', alpha=0.2)

    # Labels
    plt.text(X.min().min(), y.max().max(), '$r^{2}$ = %s' % round(r2, 2))  # r squared
    plt.text(X.min().min(), y.max().max()-10, '95% confidence $\pm$ {:.2f}'.format(abs(conf[0])))  # r squared
    plt.title(title)
    plt.ylabel('Target (y)')
    plt.xlabel('Feature (X)')
    plt.show()
