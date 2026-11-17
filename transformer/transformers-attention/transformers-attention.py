import torch

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """
    d_k=K.shape[-1]
    K=torch.transpose(K,-2,-1)
    S=torch.matmul(Q,K)/(d_k**.5)
    S=torch.softmax(S,-1)
    attention=torch.matmul(S,V)

    return attention

    