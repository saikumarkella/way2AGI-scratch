'''
    Feed Forward Network (FFN) module for the Qwen3 model.

    - Generally the FFN consists of two linear transformations with a non-linear activation in between.
    - But the Qwen3 model uses a SiLU activation function and a specific architecture for its FFN.
    - The FFN is applied to each position separately and identically.
'''

import torch
from torch import nn

class FeedForwardNetwork(nn.Module):

    def __init__(self, emd_dim, hidden_dim, dtype=None):
        super().__init__()
        self.d_in = emd_dim
        self.d_ff = hidden_dim

        # Defining the layers of the FFN
        self.linear1 = nn.Linear(self.d_in, self.d_ff, dtype=dtype, bias=False)
        self.linear2 = nn.Linear(self.d_in, self.d_ff, dtype=dtype, bias=False)
        self.linear3 = nn.Linear(self.d_ff, self.d_in, dtype=dtype, bias=False)

    def forward(self, x):
        # Forward pass through the FFN
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        # applying the SiLU activation function for x1 and pointwise multiplication with the x2
        x = torch.nn.functional.silu(x1) * x2
        x = self.linear3(x)
        return x
    
       

