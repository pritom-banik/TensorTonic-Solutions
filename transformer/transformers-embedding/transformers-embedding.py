import torch
import torch.nn as nn

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Returns an embedding layer with the requested dimensions.
    """
    embedding=nn.Embedding(vocab_size,d_model)
    return embedding

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Returns scaled token embeddings.
    """
    if not isinstance(tokens,torch.Tensor):
         tokens=torch.tensor(tokens)
        
    ans=embedding(tokens)*(d_model)**.5
    
    return ans