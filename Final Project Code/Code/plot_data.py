import util
import matplotlib.pyplot as plt

# Import data
x, y = util.load_csv('../Data/ds2_x4_x5_train.csv', add_intercept=False)

# Plot dataset
plt.figure()
plt.plot(x[y == 1, -2], x[y == 1, -1], 'ro', linewidth=2)
plt.plot(x[y == 0, -2], x[y == 0, -1], 'bo', linewidth=2)

# Add labels and save to disk
plt.xlabel('Earthquake Magnitude')
plt.ylabel('Distance to Epicenter')
plt.savefig('../Output/data.png')
