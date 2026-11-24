import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns the bottleneck residual-block output as a nested list.
    """
    s=x if Ws is None else np.matmul(x,Ws)
    
    xw1=np.maximum(np.matmul(x,W1),0)
    xw2=np.maximum(np.matmul(xw1,W2),0)
    y=np.maximum(np.matmul(xw2,W3)+s,0)

    return y