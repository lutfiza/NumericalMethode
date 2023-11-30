import numpy as np


def five_point(x, y):
    """Calculate the first derivative.

    All values in 'x' must be equally spaced.

    Args:
        x (numpy.ndarray): x values.
        y (numpy.ndarray): y values.

    Returns:
        dy (numpy.ndarray): the first derivative values.
    """
    if x.size < 6 or y.size < 6:
        raise ValueError("'x' and 'y' arrays must have 6 values or more.")

    if x.size != y.size:
        raise ValueError("'x' and 'y' must have the same size.")

    def dy_mid(h, y0, y1, y3, y4):
        return (1 / (12 * h)) * (y0 - 8 * y1 + 8 * y3 - y4)

    def dy_end(h, y0, y1, y2, y3, y4):
        return (1 / (12 * h)) * \
            (-25 * y0 + 48 * y1 - 36 * y2 + 16 * y3 - 3 * y4)

    hx = x[1] - x[0]
    n = x.size
    dy = np.zeros(n)
    for i in range(0, n):
        if i in (0, 1):
            dy[i] = dy_end(hx, y[i], y[i + 1], y[i + 2], y[i + 3], y[i + 4])
        elif i in (n - 1, n - 2):
            dy[i] = dy_end(-hx, y[i], y[i - 1], y[i - 2], y[i - 3], y[i - 4])
        else:
            dy[i] = dy_mid(hx, y[i - 2], y[i - 1], y[i + 1], y[i + 2])

    return dy

# Example 'Differentiation: Five-Point'
x = np.array([2.1, 2.2, 2.3, 2.4, 2.5, 2.6])
y = np.array([-1.709847, -1.373823, -1.119214, -0.9160143, -0.7470223, -0.6015966])

# Calculate the first derivative using five-point method
dy_five_point = five_point(x, y)

# Print the result
print("Five-Point Method - First Derivative:", dy_five_point)
