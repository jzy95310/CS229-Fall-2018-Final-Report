# Important note: you do not have to modify this file for your homework.

import util
import numpy as np
import matplotlib as plt
from sklearn.linear_model import LogisticRegression




print('==== Training model on data set A ====')

X_24, Y_24 = util.load_csv('../Data/ds1_x2_x4.csv', add_intercept=True)
theta_24 = LogisticRegression(solver='liblinear').fit(X_24, Y_24)
print(theta_24.__dict__)
theta_1 = np.array((-2.07867   , -0.06727301,  0.81436978))
np.savetxt('../Output/lr_1.txt', theta_1)
util.plot(X_24, Y_24, theta_1, 'Bridge Age', 'Earthquake Magnitude', '../Output/lr_visual_1.png')

X_25, Y_25 = util.load_csv('../Data/ds1_x2_x5.csv', add_intercept=True)
theta_25 = LogisticRegression(solver='liblinear').fit(X_25, Y_25)
print(theta_25.__dict__)
theta_2 = np.array((-0.44368227, -0.03712713, -0.00225932))
np.savetxt('../Output/lr_2.txt', theta_2)
util.plot(X_25, Y_25, theta_2, 'Bridge Age', 'Distance to Epicenter', '../Output/lr_visual_2.png')

X_45, Y_45 = util.load_csv('../Data/ds1_x4_x5.csv', add_intercept=True)
theta_45 = LogisticRegression(solver='liblinear').fit(X_45, Y_45)
print(theta_45.__dict__)
theta_3 = np.array((-2.45632541,  0.67929519, -0.0261862))
np.savetxt('../Output/lr_3.txt', theta_3)
util.plot(X_45, Y_45, theta_3, 'Earthquake Magnitude', 'Distance to Epicenter', '../Output/lr_visual_3.png')

