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

# Split labels and predictions into positive and negative classes
def split_labels(labels, predictions):
    positive_labels = []
    negative_labels = []
    positive_predictions = []
    negative_predictions = []
    m = labels.shape[0]
    for i in range(m):
        if labels[i] == 1:
            positive_labels.append(labels[i])
            positive_predictions.append(predictions[i])
        else:
            negative_labels.append(labels[i])
            negative_predictions.append(predictions[i])

    pos_labels = np.asarray(positive_labels)
    neg_labels = np.asarray(negative_labels)
    pos_predictions = np.asarray(positive_predictions)
    neg_predictions = np.asarray(negative_predictions)

    return pos_labels, neg_labels, pos_predictions, neg_predictions

# Import data
X_245_train, Y_245_train = util.load_csv('../Data/ds2_x2_x4_x5_train.csv', add_intercept=False)
params = X_245_train
labels = np.reshape(Y_245_train, (Y_245_train.shape[0], 1))
train_data = np.concatenate((params, labels), axis=1)
pos_set, neg_set = split_data(train_data)

X_245_test, Y_245_test = util.load_csv('../Data/ds2_x2_x4_x5_test.csv', add_intercept=False)

# Start training the model
iter = 100
min_pos_ratio = 0.2
max_pos_ratio = 1
n_ratio = 5
n_iter = np.linspace(1, iter, num=iter)
delta_accuracy = np.zeros((n_ratio, 2, iter))
average_accuracy = np.zeros((n_ratio, 2))
pos_accuracy = []
neg_accuracy = []

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

        lr_predict_test = lr.predict(X_245_test)
        accuracy_test = accuracy_score(Y_245_test, lr_predict_test)
        # print('The testing accuracy with pos:neg = 1:1 is: %s' % accuracy_test)
        delta_accuracy[r, 1, i] = accuracy_test

        # Check the accuracy of both positive and negative classes
        labels_pos, labels_neg, predictions_pos, predictions_neg = split_labels(Y_245_test, lr_predict_test)
        pos_accuracy.append(accuracy_score(labels_pos, predictions_pos))
        neg_accuracy.append(accuracy_score(labels_neg, predictions_neg))

    # Plot the change in accuracy with iteration number
    plt.figure()
    plt.plot(n_iter, delta_accuracy[r, 0, :], color='blue', linewidth=1.0, linestyle='-')
    plt.plot(n_iter, delta_accuracy[r, 1, :], color='red', linewidth=1.0, linestyle='-')
    plt.ylim((0.5, 1))
    plt.xlabel('# of iterations')
    plt.ylabel('Accuracy')
    plt.title('Logistic Regression with #pos/#neg = %.1f' %ratio)
    plt.savefig('../Output/log_reg_x2_x4_x5/log_reg_%.1f.png' %ratio)

    # Compute the average accuracy
    average_accuracy[r, 0] = np.average(delta_accuracy[r, 0, :])
    average_accuracy[r, 1] = np.average(delta_accuracy[r, 1, :])

np.savetxt('../Output/log_reg_x2_x4_x5/log_reg_accuracy.txt', average_accuracy)
train_accuracy = np.average(average_accuracy[:, 0])
test_accuracy = np.average(average_accuracy[:, 1])
pos_accuracy = np.asarray(pos_accuracy)
neg_accuracy = np.asarray(neg_accuracy)
positive_accuracy = np.average(pos_accuracy)
negative_accuracy = np.average(neg_accuracy)
print('Training accuracy for logistic regression is: %.4f.' %train_accuracy)
print('Testing accuracy for logistic regression is: %.4f.' %test_accuracy)
print('Testing accuracy of POSITIVE classes for logistic regression is: %.4f.' %positive_accuracy)
print('Testing accuracy of NEGATIVE classes for logistic regression is: %.4f.' %negative_accuracy)

# Part 2: Quadratic Discriminant Analysis
delta_accuracy = np.zeros((n_ratio, 2, iter))
average_accuracy = np.zeros((n_ratio, 2))
pos_accuracy = []
neg_accuracy = []

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

        qda_predict_test = qda.predict(X_245_test)
        accuracy_test = accuracy_score(Y_245_test, qda_predict_test)
        # print('The testing accuracy with pos:neg = 1:1 is: %s' % accuracy_test)
        delta_accuracy[r, 1, i] = accuracy_test

        # Check the accuracy of both positive and negative classes
        labels_pos, labels_neg, predictions_pos, predictions_neg = split_labels(Y_245_test, qda_predict_test)
        pos_accuracy.append(accuracy_score(labels_pos, predictions_pos))
        neg_accuracy.append(accuracy_score(labels_neg, predictions_neg))

    # Plot the change in accuracy with iteration number
    plt.figure()
    plt.plot(n_iter, delta_accuracy[r, 0, :], color='blue', linewidth=1.0, linestyle='-')
    plt.plot(n_iter, delta_accuracy[r, 1, :], color='red', linewidth=1.0, linestyle='-')
    plt.ylim((0.8, 1))
    plt.xlabel('# of iterations')
    plt.ylabel('Accuracy')
    plt.title('QDA with #pos/#neg = %.1f' % ratio)
    plt.savefig('../Output/QDA_x2_x4_x5/qda_%.1f.png' % ratio)

    # Compute the average accuracy
    average_accuracy[r, 0] = np.average(delta_accuracy[r, 0, :])
    average_accuracy[r, 1] = np.average(delta_accuracy[r, 1, :])

np.savetxt('../Output/QDA_x2_x4_x5/qda_accuracy.txt', average_accuracy)
train_accuracy = np.average(average_accuracy[:, 0])
test_accuracy = np.average(average_accuracy[:, 1])
pos_accuracy = np.asarray(pos_accuracy)
neg_accuracy = np.asarray(neg_accuracy)
positive_accuracy = np.average(pos_accuracy)
negative_accuracy = np.average(neg_accuracy)
print('Training accuracy for QDA is: %.4f.' %train_accuracy)
print('Testing accuracy for QDA is: %.4f.' %test_accuracy)
print('Testing accuracy of POSITIVE classes for QDA is: %.4f.' %positive_accuracy)
print('Testing accuracy of NEGATIVE classes for QDA is: %.4f.' %negative_accuracy)

# Part 3: K-Nearest Neighbors Classifier
delta_accuracy = np.zeros((n_ratio, 2, iter))
average_accuracy = np.zeros((n_ratio, 2))
pos_accuracy = []
neg_accuracy = []

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

        knn_predict_test = knn.predict(X_245_test)
        accuracy_test = accuracy_score(Y_245_test, knn_predict_test)
        # print('The testing accuracy with pos:neg = 1:1 is: %s' % accuracy_test)
        delta_accuracy[r, 1, i] = accuracy_test

        # Check the accuracy of both positive and negative classes
        labels_pos, labels_neg, predictions_pos, predictions_neg = split_labels(Y_245_test, knn_predict_test)
        pos_accuracy.append(accuracy_score(labels_pos, predictions_pos))
        neg_accuracy.append(accuracy_score(labels_neg, predictions_neg))

    # Plot the change in accuracy with iteration number
    plt.figure()
    plt.plot(n_iter, delta_accuracy[r, 0, :], color='blue', linewidth=1.0, linestyle='-')
    plt.plot(n_iter, delta_accuracy[r, 1, :], color='red', linewidth=1.0, linestyle='-')
    plt.ylim((0.5, 1))
    plt.xlabel('# of iterations')
    plt.ylabel('Accuracy')
    plt.title('KNN with #pos/#neg = %.1f' % ratio)
    plt.savefig('../Output/KNN_x2_x4_x5/knn_%.1f.png' % ratio)

    # Compute the average accuracy
    average_accuracy[r, 0] = np.average(delta_accuracy[r, 0, :])
    average_accuracy[r, 1] = np.average(delta_accuracy[r, 1, :])

np.savetxt('../Output/KNN_x2_x4_x5/knn_accuracy.txt', average_accuracy)
train_accuracy = np.average(average_accuracy[:, 0])
test_accuracy = np.average(average_accuracy[:, 1])
pos_accuracy = np.asarray(pos_accuracy)
neg_accuracy = np.asarray(neg_accuracy)
positive_accuracy = np.average(pos_accuracy)
negative_accuracy = np.average(neg_accuracy)
print('Training accuracy for KNN is: %.4f.' %train_accuracy)
print('Testing accuracy for KNN is: %.4f.' %test_accuracy)
print('Testing accuracy of POSITIVE classes for KNN is: %.4f.' %positive_accuracy)
print('Testing accuracy of NEGATIVE classes for KNN is: %.4f.' %negative_accuracy)
