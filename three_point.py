import numpy as np


def three_point(x, y):
    """Calculate the first derivative.

    All values in 'x' must be equally spaced.

    Args:
        x (numpy.ndarray): x values.
        y (numpy.ndarray): y values.

    Returns:
        dy (numpy.ndarray): the first derivative values.
    """
    if x.size < 3 or y.size < 3:
        raise ValueError("'x' and 'y' arrays must have 3 values or more.")

    if x.size != y.size:
        raise ValueError("'x' and 'y' must have the same size.")

    def dy_mid(h, y0, y2):
        return (1 / (2 * h)) * (y2 - y0)

    def dy_end(h, y0, y1, y2):
        return (1 / (2 * h)) * (-3 * y0 + 4 * y1 - y2)

    hx = x[1] - x[0]
    n = x.size
    dy = np.zeros(n)
    for i in range(0, n):
        if i == 0:
            dy[i] = dy_end(hx, y[i], y[i + 1], y[i + 2])
        elif i == n - 1:
            dy[i] = dy_end(-hx, y[i], y[i - 1], y[i - 2])
        else:
            dy[i] = dy_mid(hx, y[i - 1], y[i + 1])

    return dy

# Example 'Differentiation: Three-Point'
X = np.array([1.1, 1.2, 1.3, 1.4])
y = np.array([9.025013, 11.02318, 13.46374, 16.44465])

# Calculate the first derivative using three-point method
dy_three_point = three_point(X, y)

# Print the result
print("Three-Point Method - First Derivative:", dy_three_point)
