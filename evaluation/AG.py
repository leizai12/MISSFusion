import numpy as np


def compute_ag(image):
    """Compute average gradient."""
    width = image.shape[1] - 1
    height = image.shape[0] - 1
    grad_y, grad_x = np.gradient(image)
    gradient = np.sqrt((np.square(grad_x) + np.square(grad_y)) / 2)
    return float(np.sum(np.sum(gradient)) / (width * height))
