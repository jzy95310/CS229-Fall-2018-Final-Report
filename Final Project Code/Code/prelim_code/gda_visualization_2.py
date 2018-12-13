import numpy as np
import util

from linear_model import LinearModel


def main():
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_24, y_24 = util.load_csv('../Data/ds1_x2_x4.csv', add_intercept=False)
    Gauss_analysis = GDA()
    theta = Gauss_analysis.fit(x_24, y_24)
    print(theta)
    np.savetxt('../Output/GDA_1.txt', theta)
    util.plot(x_24, y_24, theta, 'Bridge Age', 'Earthquake Magnitude', '../Output/GDA_visual_1.png')

    x_25, y_25 = util.load_csv('../Data/ds1_x2_x5.csv', add_intercept=False)
    Gauss_analysis = GDA()
    theta = Gauss_analysis.fit(x_25, y_25)
    print(theta)
    np.savetxt('../Output/GDA_2.txt', theta)
    util.plot(x_25, y_25, theta, 'Bridge Age', 'Distance to Epicenter', '../Output/GDA_visual_2.png')

    x_45, y_45 = util.load_csv('../Data/ds1_x4_x5.csv', add_intercept=False)
    Gauss_analysis = GDA()
    theta = Gauss_analysis.fit(x_45, y_45)
    print(theta)
    np.savetxt('../Output/GDA_3.txt', theta)
    util.plot(x_45, y_45, theta, 'Earthquake Magnitude', 'Distance to Epicenter', '../Output/GDA_visual_3.png')

    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    @staticmethod
    def fit(x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        m = np.shape(x)[0]
        n = np.shape(x)[1]

        phi = 1/m*np.sum(y)

        index0 = np.where(y == 0)
        mu_0 = np.sum(x[index0], axis=0)/len(index0[0])

        index1 = np.where(y == 1)
        mu_1 = np.sum(x[index1], axis=0)/len(index1[0])

        Sigma = np.zeros((n, n))

        for i in range(m):
            if y[i] == 0:
                mu = mu_0
            else:
                mu = mu_1
            Sigma += np.outer((x[i,:] - mu).T, (x[i,:] - mu))
        Sigma = Sigma/m

        # Write theta in terms of the parameters
        SigmaInv = np.linalg.inv(Sigma)

        theta = np.dot(SigmaInv, (mu_1 - mu_0))

        theta0 = 1/2*np.dot(np.dot(mu_0.T, SigmaInv), mu_0) - 1/2*np.dot(np.dot(mu_1.T, SigmaInv), mu_1) - np.log(1 - phi) + np.log(phi)

        theta = np.insert(theta, 0, theta0)

        return theta

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
    #     output = 1/(1 + np.exp(-(np.dot(self.theta, x.T))))
    #     return output
        # *** END CODE HERE
if __name__ == '__main__':
    main()