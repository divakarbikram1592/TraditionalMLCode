from sklearn.cluster import AgglomerativeClustering

# Create sample data
data = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]

# Perform hierarchical clustering with 3 clusters
clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
clustering.fit(data)

# Print cluster labels
print(clustering.labels_)
