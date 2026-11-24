import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns the last-axis normalized array.
    """
    norm=(x-np.mean(x,axis=-1,keepdims=True,dtype=np.float64))/np.sqrt(np.add(np.var(x,axis=-1, keepdims=True,dtype=np.float64),eps,dtype=np.float64))
    y=np.add(gamma*norm,beta,dtype=np.float64)
    return y