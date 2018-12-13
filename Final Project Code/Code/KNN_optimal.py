import numpy as np
import util
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
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
X_245_train, Y_245_train = util.load_csv('../Data/KNN_optimal/ds2_knn_train.csv', add_intercept=False)
params = X_245_train
labels = np.reshape(Y_245_train, (Y_245_train.shape[0], 1))
train_data = np.concatenate((params, labels), axis=1)
pos_set, neg_set = split_data(train_data)

X_245_cv, Y_245_cv = util.load_csv('../Data/KNN_optimal/ds2_knn_cv.csv', add_intercept=False)

# Start training the model
iter = 100
min_pos_ratio = 0.2
max_pos_ratio = 1
n_ratio = 5
k_range = np.linspace(1, 10, 10)
n_iter = np.linspace(1, iter, num=iter)
delta_accuracy = np.zeros((n_ratio, 2, iter))
average_accuracy = np.zeros((n_ratio, 2))
cv_accuracy = np.zeros((len(k_range),))

# Part 3: K-Nearest Neighbors Classifier
for k in k_range:
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
            knn = KNeighborsClassifier(n_neighbors=int(k)).fit(X, y)

            # Prediction with KNN classifier
            knn_predict_train = knn.predict(X)
            accuracy_train = accuracy_score(y, knn_predict_train)
            # print('The training accuracy with pos:neg = 1:1 is: %s' % accuracy_train)
            delta_accuracy[r, 0, i] = accuracy_train

            knn_predict_cv = knn.predict(X_245_cv)
            accuracy_cv = accuracy_score(Y_245_cv, knn_predict_cv)
            # print('The testing accuracy with pos:neg = 1:1 is: %s' % accuracy_test)
            delta_accuracy[r, 1, i] = accuracy_cv

        # Compute the average accuracy
        average_accuracy[r, 0] = np.average(delta_accuracy[r, 0, :])
        average_accuracy[r, 1] = np.average(delta_accuracy[r, 1, :])

    # Compute the cross-validated accuracy for values of k from 1 to 10
    cv_accuracy[int(k)-1] = np.average(average_accuracy[:, 1])

# Plot the cross-validated accuracy with respect to the k values
plt.figure()
plt.plot(k_range, cv_accuracy, color='blue', linewidth=2.0, linestyle='-')
plt.xlabel('K value')
plt.ylabel('Cross-validation accuracy')
plt.title('K value VS. Cross-validation accuracy')
plt.savefig('../Output/KNN_optimal.png')

k_optimal = np.argmax(cv_accuracy) + 1
print('The optimal K value is %s.' %k_optimal)