import numpy as np

def conv_block(x, W1, W2, Ws):
    """
    Returns the projection residual-block output as a nested list.
    """
    layer1=np.matmul(x,W1)
    layer1=np.maximum(layer1,0)

    layer2=np.matmul(layer1,W2)+np.matmul(x,Ws)
    layer2=np.maximum(layer2,0)

    return layer2