import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Returns the gradient propagated through residual Jacobians.
    """
    g=np.asarray(x,dtype=np.float64).copy()

    for j in gradients_F:
        g=g+np.matmul(g,j)

    return g

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Returns the gradient propagated through plain Jacobians.
    """
    g=np.asarray(x,dtype=np.float64).copy()

    for j in gradients_F:
        g=np.matmul(g,j)
    return g