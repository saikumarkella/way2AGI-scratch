'''
    Positional Encodings
    - RoPE == Rotary Positional Encodings

    A Rotary Positional Encodings has a 2 kinds of implementations.
    1. Implementation from the hugging Face.
    2. Actual Multi-Dimensional implementation of RoPE ( Rotary Positional Encodings ).
'''

# importing the modules
import torch.nn as nn
import torch

# computing the rotation positional embeddings parameters
def compute_rope_params(head_dim, theta_base = 10_000, context_length = 4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # indexing dimensions
    indexes = torch.arange(0, head_dim, 2, dtype=dtype)
    # compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (indexes / head_dim))

    # getting the positions based on the context length
    positions = torch.arange(context_length).float()

    # computing the angles using the positions and inverse frequencies
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=1)

    # return the angles
    return torch.cos(angles), torch.sin(angles)



# Applying the Rotary Position Embeddings.
def apply_rope(x, cos, sin, offset=0):
    d = x.shape[-1]
    x1 = x[..., :d//2] # "first element of every pair"
    x2 = x[..., d//2:] # "second element of every pair"

    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


# Sanity check for the RoPE implementation
if __name__ == "__main__":
    # Define the parameters
    head_dim = 8
    context_length = 16

    # Compute the RoPE parameters
    cos, sin = compute_rope_params(head_dim, context_length=context_length)

    # Create a sample input tensor
    x = torch.randn(1, context_length, head_dim)

    # Apply RoPE
    x_rope = apply_rope(x, cos, sin)

    # Print the results
    print("Original Input shape : \n", x.shape)
    print("After Applying RoPE shape : \n", x_rope.shape)
