import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Initial data points
x = np.array([0, 1, 2, 3, 4])
y = np.array([1, 3, 7, 13, 21])

# Polynomial regression function
def update_poly_regression(x, y):
    degree = 3
    x = x.reshape(-1, 1)
    poly_reg = PolynomialFeatures(degree)
    X_poly = poly_reg.fit_transform(x)
    polynomial_reg = LinearRegression()
    polynomial_reg.fit(X_poly, y)
    x_line = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    X_line_poly = poly_reg.transform(x_line)
    y_line = polynomial_reg.predict(X_line_poly)
    return x_line, y_line

# Plotting function
def update_plot(x, y):
    x_line, y_line = update_poly_regression(x, y)
    scatter.set_offsets(np.c_[x, y])
    line.set_data(x_line, y_line)
    fig.canvas.draw_idle()

# Button click event
def on_button_click(event):
    global x, y
    x = np.random.rand(5) * 4
    y = np.random.rand(5) * 25
    update_plot(x, y)

# Mouse click event
def on_pick(event):
    global selected_point
    selected_point = event.ind[0]

# Mouse release event
def on_release(event):
    global x, y, selected_point
    if selected_point is not None:
        x[selected_point] = event.xdata
        y[selected_point] = event.ydata
        update_plot(x, y)
        selected_point = None

# Set up the interactive plot
fig, ax = plt.subplots()
x_line, y_line = update_poly_regression(x, y)
scatter = ax.scatter(x, y, picker=True, s=100, edgecolor='k', c='r', alpha=0.75)
line, = ax.plot(x_line, y_line, 'b-')
ax.set_xlim(-1, 5)
ax.set_ylim(-5, 30)

# Add a button for randomizing the data points
button_ax = plt.axes([0.7, 0.01, 0.2, 0.075])
button = Button(button_ax, 'Randomize Points')
button.on_clicked(on_button_click)

# Connect events
selected_point = None
fig.canvas.mpl_connect('pick_event', on_pick)
fig.canvas.mpl_connect('button_release_event', on_release)

# Show the interactive plot
plt.show()
