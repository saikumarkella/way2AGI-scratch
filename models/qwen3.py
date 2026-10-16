"""
    Qwen model orchestration
    - Base model of the Qwen (owned by Alibaba)
    - Here we are using the qwen3 0.6B model.
"""
import torch
from torch import nn
from networks.attentions import GroupedQueryAttention
from networks.norms import RMSNorm
from networks.ffn import FeedForwardNetwork
from networks.encodings import compute_rope_params

class Qwen3ForTextGeneration(nn.Module):
    '''
        Qwen3 Model for Text Generation. A full Model Orchestration.
        - Consists of a stack of Transformer Blocks.
    '''
    def __init__(self, config):
        super().__init__()

        # initialization of components
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size, dtype=config.dtype)

        # initializing the transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])

        # final normalization layer
        self.final_norm = RMSNorm(config.hidden_size, eps=1e-6)
        self.out_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=config.dtype)

        # initializing the utilities like rope parameters and also kv cache
        if config.head_dim is None:
            assert config.hidden_size % config.num_attention_heads == 0, "hidden size must be divisible by the number of attention heads"
            config.head_dim = config.hidden_size // config.num_attention_heads

        else:
            head_dim = config.head_dim

        cos, sin = compute_rope_params(head_dim, context_length=config.max_position_embeddings, dtype=config.dtype)

        self.re


        


class TransformerBlock(nn.Module):

    '''
        Transformer Block for the Qwen3 Model.
        - Each block consists of a Grouped Query Attention and a Feed Forward Network (FFN
        - Each block also has two RMS Normalization layers.
        - 2 residual connections.
    '''
    def __init__(self, config):
        super().__init__()
        self.attention = GroupedQueryAttention(
            d_in=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_groups=config.num_kv_groups,
            head_dim=config.head_dim,
            qk_norm=config.qk_norm,
            dtype=config.dtype
        )
        # Every block madeup of 2 RMS Normalization 
        self.norm1 = RMSNorm(config.hidden_size, eps=1e-6) 
        self.norm2 = RMSNorm(config.hidden_size, eps=1e-6)

        # Here feed forward network, the activation is SiLU
        self.feed_forward = FeedForwardNetwork(
            emd_dim=config.hidden_size,
            hidden_dim=config.intermediate_size,
            dtype=config.dtype
        )


    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        # Step 1: Apply the RMS Normalization to the input.
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.attention(x, mask, cos, sin, start_pos=start_pos, cache=cache) 
        x = x + shortcut  # Residual connection

        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = x + shortcut  # Residual connection

        return x, next_cache




