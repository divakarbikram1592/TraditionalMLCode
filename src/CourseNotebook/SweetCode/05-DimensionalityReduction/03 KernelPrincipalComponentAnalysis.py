from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons

# Generate sample data (non-linearly separable)
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

# Create a Kernel PCA object with an RBF kernel
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10)

# Fit the model to the data
kpca.fit(X)

# Transform the data to the lower-dimensional space
X_kpca = kpca.transform(X)

# Visualize the transformed data
import matplotlib.pyplot as plt

plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y)
plt.title("Kernel PCA-transformed data")
plt.show()
