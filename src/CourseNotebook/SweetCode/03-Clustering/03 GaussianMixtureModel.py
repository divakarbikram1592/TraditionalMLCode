from sklearn.mixture import GaussianMixture

# Sample data
data = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [12, 13], [14, 15]]

# Create a GMM with 3 components
gmm = GaussianMixture(n_components=3)

# Fit the GMM to the data
gmm.fit(data)

# Print the mean and covariance of each component
print("Means:")
print(gmm.means_)
print("Covariances:")
print(gmm.covariances_)

# Predict cluster labels for new data
new_data = [[11, 12], [15, 16]]
labels = gmm.predict(new_data)
print("Cluster labels for new data:", labels)

# Plot the data and fitted components
import matplotlib.pyplot as plt

plt.scatter(data[:, 0], data[:, 1], c=labels)
for i, mean in enumerate(gmm.means_):
    plt.scatter(mean[0], mean[1], marker='o', color='k', label=f"Component {i}")
plt.legend()
plt.show()
