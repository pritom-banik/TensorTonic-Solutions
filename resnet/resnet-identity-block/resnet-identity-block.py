import numpy as np

def identity_block(x, W1, W2):
    """
    Returns the identity residual-block output as a nested list.
    """
    W1=np.swapaxes(W1,0,1)
    W2=np.swapaxes(W2,0,1)
    layer1=np.matmul(x,W1)
    layer1=np.maximum(layer1,0)

    layer2=np.matmul(layer1,W2)+x
    layer2=np.maximum(layer2,0)

    return layer2