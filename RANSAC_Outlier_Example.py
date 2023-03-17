import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generate synthetic data with outliers
np.random.seed(42)
x = np.random.uniform(0, 10, size=(50, 1))
y = 2 * x ** 3 - 5 * x ** 2 + 3 * x + 10 + np.random.normal(0, 100, size=(50, 1))
y[25] += 2000  # Add an outlier
y[40] -= 2000  # Add an outlier

# Fit polynomial regression models
degree = 3
poly_features = PolynomialFeatures(degree)
X_poly = poly_features.fit_transform(x)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

ransac_poly_reg = RANSACRegressor(LinearRegression())
ransac_poly_reg.fit(X_poly, y)

# Generate prediction data
x_pred = np.linspace(0, 10, 100).reshape(-1, 1)
X_poly_pred = poly_features.transform(x_pred)

y_pred_poly = poly_reg.predict(X_poly_pred)
y_pred_ransac_poly = ransac_poly_reg.predict(X_poly_pred)

# Plot training data and predictions
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label="Training Data", color="blue", s=50)
plt.plot(x_pred, y_pred_poly, label="Polynomial Regression", color="red", linewidth=2)
plt.plot(x_pred, y_pred_ransac_poly, label="RANSAC Polynomial Regression", color="green", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparison of Polynomial Regression and RANSAC-based Polynomial Regression")
plt.legend()
plt.show()
