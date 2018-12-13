import numpy as np
import util
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

# Split data into positive and negative classes
def split_data(data_set):
    positive_set = []
    negative_set = []
    m, n = data_set.shape
    for i in range(m):
        if data_set[i, n-1] == 1:
            positive_set.append(data_set[i, :])
        else:
            negative_set.append(data_set[i, :])

    positive = np.asarray(positive_set)
    negative = np.asarray(negative_set)

    return positive, negative

# Import data
X_45_train, Y_45_train = util.load_csv('../Data/ds2_x4_x5_train.csv', add_intercept=False)
params = X_45_train
labels = np.reshape(Y_45_train, (Y_45_train.shape[0], 1))
train_data = np.concatenate((params, labels), axis=1)
pos_set, neg_set = split_data(train_data)

X_45_test, Y_45_test = util.load_csv('../Data/ds2_x4_x5_test.csv', add_intercept=False)

# Start training the model
iter = 100
min_pos_ratio = 0.2
max_pos_ratio = 1
n_ratio = 5
n_iter = np.linspace(1, iter, num=iter)
delta_accuracy = np.zeros((n_ratio, 2, iter))
average_accuracy = np.zeros((n_ratio, 2))

# Part 1: Logistic Regression
for ratio in np.linspace(min_pos_ratio, max_pos_ratio, num=n_ratio):
    r = int(round(ratio * n_ratio) - 1)
    for i in range(iter):
        # Boostrap resampling
        n_negative = neg_set.shape[0]
        n_samples = int(round(ratio * n_negative))
        pos_set_updated = resample(pos_set, replace=True, n_samples=n_samples)
        data_updated = np.concatenate((pos_set_updated, neg_set), axis=0)

        # Logistic Regression
        m, n = data_updated.shape
        X = data_updated[:, 0:n - 1]
        y = data_updated[:, n - 1]
        lr = LogisticRegression(solver='liblinear').fit(X, y)

        # Prediction with logistic regression
        lr_predict_train = lr.predict(X)
        accuracy_train = accuracy_score(y, lr_predict_train)
        # print('The training accuracy with pos:neg = 1:1 is: %s' % accuracy_train)
        delta_accuracy[r, 0, i] = accuracy_train

        lr_predict_test = lr.predict(X_45_test)
        accuracy_test = accuracy_score(Y_45_test, lr_predict_test)
        # print('The testing accuracy with pos:neg = 1:1 is: %s' % accuracy_test)
        delta_accuracy[r, 1, i] = accuracy_test

    # Plot the change in accuracy with iteration number
    plt.figure()
    plt.plot(n_iter, delta_accuracy[r, 0, :], color='blue', linewidth=1.0, linestyle='-')
    plt.plot(n_iter, delta_accuracy[r, 1, :], color='red', linewidth=1.0, linestyle='-')
    plt.ylim((0.5, 1))
    plt.xlabel('# of iterations')
    plt.ylabel('Accuracy')
    plt.savefig('../Output/log_reg_x4_x5/log_reg_%.1f.png' %ratio)

    # Compute the average accuracy
    average_accuracy[r, 0] = np.average(delta_accuracy[r, 0, :])
    average_accuracy[r, 1] = np.average(delta_accuracy[r, 1, :])

np.savetxt('../Output/log_reg_x4_x5/log_reg_accuracy.txt', average_accuracy)


# Part 2: Quadratic Discriminant Analysis
delta_accuracy = np.zeros((n_ratio, 2, iter))
average_accuracy = np.zeros((n_ratio, 2))

for ratio in np.linspace(min_pos_ratio, max_pos_ratio, num=n_ratio):
    r = int(round(ratio * n_ratio) - 1)
    for i in range(iter):
        # Boostrap resampling
        n_negative = neg_set.shape[0]
        n_samples = int(round(ratio * n_negative))
        pos_set_updated = resample(pos_set, replace=True, n_samples=n_samples)
        data_updated = np.concatenate((pos_set_updated, neg_set), axis=0)

        # Gaussian Discriminant Analysis
        m, n = data_updated.shape
        X = data_updated[:, 0:n - 1]
        y = data_updated[:, n - 1]
        qda = QuadraticDiscriminantAnalysis().fit(X, y)

        # Prediction with QDA
        qda_predict_train = qda.predict(X)
        accuracy_train = accuracy_score(y, qda_predict_train)
        # print('The training accuracy with pos:neg = 1:1 is: %s' % accuracy_train)
        delta_accuracy[r, 0, i] = accuracy_train

        qda_predict_test = qda.predict(X_45_test)
        accuracy_test = accuracy_score(Y_45_test, qda_predict_test)
        # print('The testing accuracy with pos:neg = 1:1 is: %s' % accuracy_test)
        delta_accuracy[r, 1, i] = accuracy_test

    # Plot the change in accuracy with iteration number
    plt.figure()
    plt.plot(n_iter, delta_accuracy[r, 0, :], color='blue', linewidth=1.0, linestyle='-')
    plt.plot(n_iter, delta_accuracy[r, 1, :], color='red', linewidth=1.0, linestyle='-')
    plt.ylim((0.5, 1))
    plt.xlabel('# of iterations')
    plt.ylabel('Accuracy')
    plt.savefig('../Output/QDA_x4_x5/qda_%.1f.png' % ratio)

    # Compute the average accuracy
    average_accuracy[r, 0] = np.average(delta_accuracy[r, 0, :])
    average_accuracy[r, 1] = np.average(delta_accuracy[r, 1, :])

np.savetxt('../Output/QDA_x4_x5/qda_accuracy.txt', average_accuracy)

# Part 3: K-Nearest Neighbors Classifier
delta_accuracy = np.zeros((n_ratio, 2, iter))
average_accuracy = np.zeros((n_ratio, 2))

for ratio in np.linspace(min_pos_ratio, max_pos_ratio, num=n_ratio):
    r = int(round(ratio * n_ratio) - 1)
    for i in range(iter):
        # Boostrap resampling
        n_negative = neg_set.shape[0]
        n_samples = int(round(ratio * n_negative))
        pos_set_updated = resample(pos_set, replace=True, n_samples=n_samples)
        data_updated = np.concatenate((pos_set_updated, neg_set), axis=0)

        # KNN classifier
        m, n = data_updated.shape
        X = data_updated[:, 0:n - 1]
        y = data_updated[:, n - 1]
        knn = KNeighborsClassifier(n_neighbors=2).fit(X, y)

        # Prediction with KNN classifier
        knn_predict_train = knn.predict(X)
        accuracy_train = accuracy_score(y, knn_predict_train)
        # print('The training accuracy with pos:neg = 1:1 is: %s' % accuracy_train)
        delta_accuracy[r, 0, i] = accuracy_train

        knn_predict_test = knn.predict(X_45_test)
        accuracy_test = accuracy_score(Y_45_test, knn_predict_test)
        # print('The testing accuracy with pos:neg = 1:1 is: %s' % accuracy_test)
        delta_accuracy[r, 1, i] = accuracy_test

    # Plot the change in accuracy with iteration number
    plt.figure()
    plt.plot(n_iter, delta_accuracy[r, 0, :], color='blue', linewidth=1.0, linestyle='-')
    plt.plot(n_iter, delta_accuracy[r, 1, :], color='red', linewidth=1.0, linestyle='-')
    plt.ylim((0.5, 1))
    plt.xlabel('# of iterations')
    plt.ylabel('Accuracy')
    plt.savefig('../Output/KNN_x4_x5/knn_%.1f.png' % ratio)

    # Compute the average accuracy
    average_accuracy[r, 0] = np.average(delta_accuracy[r, 0, :])
    average_accuracy[r, 1] = np.average(delta_accuracy[r, 1, :])

np.savetxt('../Output/KNN_x4_x5/knn_accuracy.txt', average_accuracy)
train_accuracy = np.average(average_accuracy[:, 0])
test_accuracy = np.average(average_accuracy[:, 1])
print('Training accuracy is: %.4f.' %train_accuracy)
print('Testing accuracy is: %.4f.' %test_accuracy)