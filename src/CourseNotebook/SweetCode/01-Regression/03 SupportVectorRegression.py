from sklearn.svm import SVR

# Sample data
X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
y = [2, 5, 7, 8, 11]

# Create an SVR model with epsilon-SVR kernel and C=100
svr = SVR(kernel='linear', C=100)

# Fit the model to the data
svr.fit(X, y)

# Make predictions for new data
new_data = [[6.5, 7.2]]
predictions = svr.predict(new_data)

# Print the predictions
print("Predictions:", predictions)

# Plot the data and fitted SVR
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.plot(X[:, 0], svr.predict(X), color='r', label='SVR')
plt.legend()
plt.show()
