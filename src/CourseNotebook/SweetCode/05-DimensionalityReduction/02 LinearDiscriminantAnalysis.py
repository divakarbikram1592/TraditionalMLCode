from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Sample data
X = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]]
y = [0, 0, 0, 1, 1, 1]  # Two classes

# Create an LDA object
lda = LinearDiscriminantAnalysis()

# Fit the LDA model to the data
lda.fit(X, y)

# Transform the data to the lower-dimensional space
transformed_data = lda.transform(X)

# Print the transformed data
print("Transformed data:")
print(transformed_data)

# Explained variance ratio (not directly applicable to LDA)
# print("Explained variance ratio:", lda.explained_variance_ratio_)

# Visualize the data in the lower-dimensional space
import matplotlib.pyplot as plt

plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=y)
plt.title("LDA-transformed data")
plt.show()
