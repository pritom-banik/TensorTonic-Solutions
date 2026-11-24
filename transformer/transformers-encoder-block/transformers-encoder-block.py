import numpy as np

def softmax(x, axis=-1):
    """Provided: numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns x normalized over the last axis, then scaled by gamma and shifted by beta.
    """
    mean=np.mean(x,dtype=np.float64,axis=-1,keepdims=True)
    variance=np.var(x,dtype=np.float64,axis=-1,keepdims=True)

    norm=(x-mean)/np.sqrt(variance+eps)
    norm=np.add(gamma*norm,beta,dtype=np.float64)
    return norm

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns multi-head attention of Q, K, V, projecting them with W_q, W_k, W_v and the merged heads with W_o.
    """
    batch_siz,seq,dim=K.shape
    d_k=dim//num_heads
    
    q=np.matmul(Q,W_q)
    k=np.matmul(K,W_k)
    v=np.matmul(V,W_v)

    q=q.reshape(batch_siz,seq,num_heads,d_k)
    k=k.reshape(batch_siz,seq,num_heads,d_k)
    v=v.reshape(batch_siz,seq,num_heads,d_k)

    q=np.swapaxes(q,1,2)
    k=np.swapaxes(k,1,2)
    v=np.swapaxes(v,1,2)

    kt=np.swapaxes(k,-2,-1)

    val=np.matmul(q,kt)/np.sqrt(d_k)

    s=softmax(val)

    head=np.matmul(s,v)
    head=np.swapaxes(head,1,2)
    head=head.reshape(batch_siz,seq,dim)
    
    head=np.matmul(head,W_o)
    return head

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Returns the position-wise feed-forward network output with a ReLU hidden layer.
    """
    h=np.add(np.matmul(x,W1),b1)
    h=np.maximum(h,0)
    fnn=np.add(np.matmul(h,W2),b2)

    return fnn

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                  W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray,
                  beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray,
                  num_heads: int) -> np.ndarray:
    """
    Returns the post-normalized Transformer encoder states.
    """
    mha=multi_head_attention(x,x,x,W_q,W_k,W_v,W_o,num_heads)
    val_z=x+mha
    z=layer_norm(val_z,gamma1,beta1)

    val_y=z+feed_forward(z,W1,b1,W2,b2)
    y=layer_norm(val_y,gamma2,beta2)

    return y
    
