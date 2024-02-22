from sklearn.decomposition import PCA

# Sample data
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

# Create a PCA object with 2 components
pca = PCA(n_components=2)

# Fit the PCA model to the data
pca.fit(data)

# Transform the data to the lower-dimensional space
transformed_data = pca.transform(data)

# Print the transformed data
print("Transformed data:")
print(transformed_data)

# Explained variance ratio
print("Explained variance ratio:", pca.explained_variance_ratio_)

# Visualize the data in the lower-dimensional space
import matplotlib.pyplot as plt

plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
plt.title("PCA-transformed data")
plt.show()
