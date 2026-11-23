import numpy as np

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    batch_size,seq_len,dimension=K.shape
    d_k=dimension//num_heads
    
    #getting q,k,d
    q=np.matmul(Q,W_q)
    k=np.matmul(K,W_k)
    v=np.matmul(V,W_v)
    
    #devide heads
    q=q.reshape(batch_size,seq_len,num_heads,d_k)
    k=k.reshape(batch_size,seq_len,num_heads,d_k)
    v=v.reshape(batch_size,seq_len,num_heads,d_k)

    #dim,heads,seq,dimension
    q=np.swapaxes(q,1,2)
    k=np.swapaxes(k,1,2)
    v=np.swapaxes(v,1,2)

    kt=np.swapaxes(k,-2,-1)
    
    scores=np.matmul(q,kt)/np.sqrt(d_k)
    
    #softmax
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    s = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
    
    head=np.matmul(s,v)
    head=np.swapaxes(head,1,2)

    head=head.reshape(batch_size,seq_len,dimension)
    mha=np.matmul(head,W_o)

    return mha
        