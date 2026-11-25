import numpy as np

def batch_norm(x,gemma,beta):
    mean=np.mean(x,axis=0,keepdims=True,dtype=np.float64)
    var=np.var(x,axis=0,keepdims=True,dtype=np.float64)

    norm=(x-mean)/np.sqrt(var+1e-5)
    scale=(norm*gemma)+beta

    return scale
    

def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns the normalized residual-block result and selected mode in a dictionary.
    """
    if mode=="post":
        linear=np.matmul(x,W1)
        norm1=batch_norm(linear,gamma1,beta1)
        relu1=np.maximum(norm1,0)
        linear2=np.matmul(relu1,W2)
        norm2=batch_norm(linear2,gamma2,beta2)
        y=np.maximum(norm2+x,0)

    else:
        norm1=batch_norm(x,gamma1,beta1)
        relu1=np.maximum(norm1,0)
        linear=np.matmul(relu1,W1)
        norm2=batch_norm(linear,gamma2,beta2)
        relu2=np.maximum(norm2,0)
        y=np.matmul(relu2,W2)
        y=y+x


    return {
        "output":y,
        "mode":mode
    }
    