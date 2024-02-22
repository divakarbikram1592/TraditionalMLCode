import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Sample data
x = np.array([[1], [2], [3], [4], [5]]).reshape((-1, 1))
y = np.array([2, 4, 5, 4, 5])

# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(x)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_poly, y)

# Print coefficients
print("Coefficients:", model.coef_[0])

# Predict for new data points
new_x = np.array([[2.5], [4.8]]).reshape((-1, 1))
predictions = model.predict(poly.transform(new_x))

# Print predictions
print("Predictions:", predictions)
