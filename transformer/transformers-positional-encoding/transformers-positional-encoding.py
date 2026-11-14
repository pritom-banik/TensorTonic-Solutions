import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    ans=np.empty((seq_length,d_model))
    for position in range(seq_length):
        for idx in range(d_model):
            i=idx//2
            down=10000**(2*i/d_model)
            if idx%2==0:
                ans[position,idx]=np.sin(position/down)
            else:
                ans[position,idx]=np.cos(position/down)
        
    return ans
        