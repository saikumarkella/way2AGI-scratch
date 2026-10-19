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

        # registering the cos and sin as buffers
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.cfg = config
        self.current_pos = 0 # Tracking the Current Position in KV Cache.


    def forward(self, in_idx, cache = None):
        '''
            Step 1: Token Embedding
            Step 2: Cache
            Step 3: Transformer block (will do the RoPE. QKNorm)
            Step 4: Final RMSNorm 
            Step 5: Linear output layer.
        '''

        # Step 1 : Embedding the tokens
        tok_emb = self.token_embedding(in_idx) # shape = (B, num_tokens, token_dim)

        x = tok_emb 
        # Step 2 : Caching 
        num_tokens = x.shape[1]
        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            self.current_pos = pos_end
            # will create a mask here
            mask = torch.triu(
                torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool),
                diagonal=1
            )[pos_start:pos_end, :pos_end]

        else:
            pos_start = 0
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool),
                diagonal=1
            )

        mask = mask[None, None, :, :] # dimens ==> Preparing for the Multi-head attention computing mask

        # transformer blocks 
        for i, block in enumerate(self.transformer_blocks):
            blk_cache =cache.get(i) # getting the cache at the layer wise
            x, new_blk_cache = block(x,mask, self.cos, self.sin, start_pos = pos_start, cache = blk_cache)

            if cache is not None:
                cache.update(i, new_blk_cache)

        # Computing the Final Normalization
        x = self.final_norm(x)

        # Computing the output layer.
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits

    def reset_kv_cache(self):
        self.current_pos = 0




        


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




