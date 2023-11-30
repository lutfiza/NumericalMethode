import numpy as np


def backward_difference(x, y):
    """Calculate the first derivative.

    All values in 'x' must be equally spaced.

    Args:
        x (numpy.ndarray): x values.
        y (numpy.ndarray): y values.

    Returns:
        dy (numpy.ndarray): the first derivative values.
    """
    if x.size < 2 or y.size < 2:
        raise ValueError("'x' and 'y' arrays must have 2 values or more.")

    if x.size != y.size:
        raise ValueError("'x' and 'y' must have the same size.")

    def dy_difference(h, y0, y1):
        return (y1 - y0) / h

    n = x.size
    dy = np.zeros(n)
    for i in range(0, n):
        if i == n - 1:
            hx = x[i] - x[i - 1]
            dy[i] = dy_difference(-hx, y[i], y[i - 1])
        else:
            hx = x[i + 1] - x[i]
            dy[i] = dy_difference(hx, y[i], y[i + 1])

    return dy

# Example 'Differentiation: Backward-difference'
X = np.array([0.0, 0.2, 0.4])
y = np.array([0.00000, 0.74140, 1.3718])

# Calculate the first derivative using backward difference
dy_backward = backward_difference(X, y)

# Print the result
print("Backward Difference - First Derivative:", dy_backward)