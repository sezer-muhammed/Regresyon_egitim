import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generate synthetic data
np.random.seed(42)
x_train = np.random.uniform(0, 2 * np.pi, size=(20, 1))
y_train = np.sin(x_train) + np.random.normal(0, 0.1, size=(20, 1))

# Fit polynomial regression
degree = 3
poly_reg = PolynomialFeatures(degree)
X_poly_train = poly_reg.fit_transform(x_train)
polynomial_reg = LinearRegression()
polynomial_reg.fit(X_poly_train, y_train)

# Generate prediction data
x_pred = np.linspace(-1 * np.pi, 3 * np.pi, 100).reshape(-1, 1)
X_poly_pred = poly_reg.transform(x_pred)
y_pred = polynomial_reg.predict(X_poly_pred)

# Plot training data and predictions
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, label="Training Data", color="blue", s=50)
plt.plot(x_pred, y_pred, label="Polynomial Regression (degree 3)", color="red", linewidth=2)
plt.axvline(x=0, linestyle="--", color="green", label="x-axis range of training data")
plt.axvline(x=2 * np.pi, linestyle="--", color="green")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Limitations of Polynomial Regression: Extrapolation")
plt.legend()
plt.show()
