import numpy as np

def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    """
    Returns the network logits as a nested list.
    """
    h=np.maximum(np.matmul(x,conv1),0)

    y1=np.maximum(np.matmul(np.maximum(np.matmul(h,W1_b1),0),W2_b1)+h,0)
    y2=np.maximum(np.matmul(np.maximum(np.matmul(y1,W1_b2),0),W2_b2)+np.matmul(y1,Ws_b2),0)
    logits=np.matmul(y2,fc)
    return logits
    