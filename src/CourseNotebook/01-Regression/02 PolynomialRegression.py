import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generate some sample data
np.random.seed(0)
x = 2 - 3 * np.random.normal(0, 1, 20)
y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 20)

# Transform the data to include another axis to fit the model
x = x[:, np.newaxis]
y = y[:, np.newaxis]

# Polynomial regression
polynomial_features= PolynomialFeatures(degree=3)
x_poly = polynomial_features.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

# Plotting the actual data and the polynomial fit
plt.scatter(x, y, s=10)
plt.plot(x, y_poly_pred, color='r')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.show()
