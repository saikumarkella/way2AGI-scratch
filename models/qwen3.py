"""
    Qwen model orchestration
    - Base model of the Qwen (owned by Alibaba)
    - Here we are using the qwen3 0.6B model.
"""
import torch
from torch import nn
from networks.attentions import GroupedQueryAttention
from networks.norms import RMSNorm


class Qwen3ForTextGeneration(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Defining the model architecture using the configurations

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            GroupedQueryAttention(
                d_in=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_groups=config.num_kv_groups,
                head_dim=config.head_dim,
                qk_norm=config.qk_norm,
                dtype=config.dtype
            ) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=1e-6)
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, x):
        # step 1: Embedding the input tokens
        # step 2: skip the embeddings
        # step 3: Normalization using RMS Normalization - 1
        # step 4:  
        pass


    def load_pretrained_weights(self, weights_path):
        # Load the pretrained weights from the specified path
        state_dict = torch.load(weights_path)
        self.load_state_dict(state_dict)

        


# Defining the transformer Block
class TransformerBlock(nn.Module):

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
        self.norm1 = RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = RMSNorm(config.hidden_size, eps=1e-6)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )

    def forward(self, x, mask=None):
        pass

