import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Read data from CSV file
df = pd.read_csv('BostonHousing.csv')

# Separate features and target
X = df.drop('medv', axis=1)
y = df['medv']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Visualize the effect of each parameter
fig, axes = plt.subplots(5, 3, figsize=(15, 20))
for idx, column in enumerate(X.columns):
    ax = axes.flatten()[idx]
    ax.scatter(X_train[column], y_train, color='blue', alpha=0.5, label='Train data')
    ax.scatter(X_test[column], y_test, color='red', alpha=0.5, label='Test data')
    ax.set_xlabel(column)
    ax.set_ylabel('medv')
    ax.legend()

# Remove the last empty subplot
axes.flatten()[-1].axis('off')

# Add a graph for output vs. predictions
ax_pred = axes.flatten()[-2]
ax_pred.scatter(y_train, y_train_pred, color='blue', alpha=0.5, label='Train data')
ax_pred.scatter(y_test, y_test_pred, color='red', alpha=0.5, label='Test data')
ax_pred.set_xlabel('True values')
ax_pred.set_ylabel('Predicted values')
ax_pred.set_title('Output vs. Predictions')
ax_pred.legend()

plt.tight_layout()
plt.show()
