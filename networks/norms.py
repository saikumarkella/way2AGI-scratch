"""
    Defining the Normalizations
    - RMS Normalization (This will leads to stablize the training)

    Note : In future different normalization can be added here.
"""

# importing the modules
import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(self, emb_dim, eps = 1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shit = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).means(dim=-1, keepdim = True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shit is not None:
            norm_x = norm_x + self.shit

        return norm_x.to(input_dtype)




