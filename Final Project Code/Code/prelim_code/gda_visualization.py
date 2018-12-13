import numpy as np
import util

from linear_model import LinearModel


def sigmoid(z):
    """Sigmoid function"""
    g = 1/(1+np.exp(-z))
    return g


def main():
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_45, y_45 = util.load_csv('../Data/ds1_x4_x5.csv', add_intercept=True)
    Gauss_analysis = GDA()
    theta = Gauss_analysis.fit(x_45, y_45)
    print(theta)
    np.savetxt('../Output/GDA_3.txt', theta)
    util.plot(x_45, y_45, theta, '../Output/GDA_visual_3.png')




    # *** START CODE HERE ***
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to pred_path
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters
        m = x.shape[0]
        n = x.shape[1]
        y = y.reshape(m, 1)

        y_neg = 1 - y
        theta = 1 / m * np.sum(y)
        sum_0, sum_1 = np.zeros((n, 1))
        for i in range(m):
            if y_neg[i] == 1:
                sum_0 = sum_0 + np.transpose(x[i, :])
            else:
                sum_1 = sum_1 + np.transpose(x[i, :])
        mu_0 = sum_0 / np.sum(y_neg)
        mu_1 = sum_1 / np.sum(y)
        mu_0 = mu_0.reshape(n, 1)
        mu_1 = mu_1.reshape(n, 1)
        sum_sigma = np.zeros((n, n))
        for i in range(m):
            if y_neg[i] == 1:
                diff = np.transpose(x[i, :].reshape(1, n)) - mu_0
            else:
                diff = np.transpose(x[i, :].reshape(1, n)) - mu_1
            sum_sigma = sum_sigma + np.outer(diff, diff)
        sigma = sum_sigma / m
        theta_main = np.matmul(np.transpose(mu_1 - mu_0), np.linalg.inv(sigma))
        theta_main = np.transpose(theta_main)
        theta_0 = 0.5 * np.matmul(np.matmul(mu_0.T, np.linalg.inv(sigma)), mu_0) - 0.5 * np.matmul(np.matmul(mu_1.T, np.linalg.inv(sigma)), mu_1) + np.log(theta) - np.log(1 - theta)
        theta = np.concatenate((theta_0, theta_main), axis = 0)
        self.theta = theta
        print(self.theta)
        return self.theta

        # *** END CODE HERE ***

    # def predict(self, x):
    #     """Make a prediction given new inputs x.
    #
    #     Args:
    #         x: Inputs of shape (m, n).
    #
    #     Returns:
    #         Outputs of shape (m,).
    #     """
    #     # *** START CODE HERE ***
    #     theta_main = self.theta[1:]
    #     theta_0 = self.theta[0]
    #     eta = np.matmul(x, theta_main) + theta_0
    #     p = sigmoid(eta)
    #     return p
    #     # *** END CODE HERE
if __name__ == '__main__':
    main()