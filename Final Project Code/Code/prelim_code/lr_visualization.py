# Important note: you do not have to modify this file for your homework.

import util
import numpy as np
import matplotlib as plt


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    m, n = X.shape

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))

    return grad


def logistic_regression(X, Y):
    """Train a logistic regression model."""
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 1e-1
    Lambda = 0.05

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta) + Lambda/(m)*theta
        theta = theta - learning_rate * grad
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
            print(np.linalg.norm(prev_theta - theta))
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    print(theta)
    return theta


def main():
    print('==== Training model on data set A ====')

    X_24, Y_24 = util.load_csv('../Data/ds1_x2_x4.csv', add_intercept=True)
    theta = logistic_regression(X_24, Y_24)
    np.savetxt('../Output/lr_1.txt', theta)
    util.plot(X_24, Y_24, theta, 'Bridge Age', 'Earthquake Magnitude', '../Output/lr_visual_1.png')

    X_25, Y_25 = util.load_csv('../Data/ds1_x2_x5.csv', add_intercept=True)
    theta = logistic_regression(X_25, Y_25)
    np.savetxt('../Output/lr_2.txt', theta)
    util.plot(X_25, Y_25, theta, 'Bridge Age', 'Distance to Epicenter', '../Output/lr_visual_2.png')

    X_45, Y_45 = util.load_csv('../Data/ds1_x4_x5.csv', add_intercept=True)
    theta = logistic_regression(X_45, Y_45)
    np.savetxt('../Output/lr_3.txt', theta)
    util.plot(X_45, Y_45, theta, 'Earthquake Magnitude', 'Distance to Epicenter', '../Output/lr_visual_3.png')
    # print(Xa)

    # print('\n==== Training model on data set B ====')
    # Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    # logistic_regression(Xb, Yb)


if __name__ == '__main__':
    main()
