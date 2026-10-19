'''
   Grouped query Attention Mechanism. 

   - Grouped Query Attention >> 
'''
import torch
from torch import nn
from networks.norms import RMSNorm
from networks.encodings import apply_rope, compute_rope_params
class GroupedQueryAttention(nn.Module):

    def __init__(self, d_in, num_heads, num_kv_groups, head_dim = None, qk_norm = False, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "Number of must divisible by num of kv groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = self.num_heads // self.num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "d_in must divisible by the number heads"
            head_dim = d_in // num_heads

        # configuring the parameters
        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        # Defining the weights
        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype = dtype)  # defining the weighted query
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype) # defining the weighted key
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype = dtype)  # Defining the weighted Value

        # output projection
        self.out_proj = nn.Linear()

        # Normaization
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=1e-6)  # need to define this RMS normalization
            self.k_norm = RMSNorm(self.head_dim, eps=1e-6)  # Need to Define the RMS Normalization
        else:
            self.q_norm = self.k_norm = None  # No Normalization happens here.

                
    def forward(self, x,mask, cos,sin, start_pos = 0, cache =None):
        b, num_tokens, _ = x.shape

        # Applying the projections
        queries = self.W_query(x) # (batch, num_tokens, num_heads * head_dim)
        keys = self.W_key(x) # (batch, num_tokens, num_kv_heads * head_dim)
        values = self.W_value(x) # (batch, num_tokens, num_kv_heads * head_dim)

        # Reshape to heads / kv-groups
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1,2)
        keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1,2)
        values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1,2)

        # Applying the RMS Normalization
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys_new = self.k_norm(keys_new)

        # applying the the RoPE to the queries and keys only.
        queries = apply_rope(queries, cos, sin , offset=start_pos)
        keys_new = apply_rope(keys_new, cos,sin, offset=start_pos)

        # Defining the KV cache.
        if cache is not None:
            prev_k, prev_v = cache
            keys = torch.cat([prev_k, keys_new], dim=2)
            values = torch.cat([prev_v, values_new], dim=2)
        else:
            start_pos = 0  # resetting the RoPE positioning
            keys, values = keys_new, values_new
        next_cache = (keys, values)


        # Expanding the K, V to match number of heads.
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Computing the Attentions
        attn_scores = queries @ keys.transpose(2,3)
        attn_scores = attn_scores.masked_fill(mask,-torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5 , dim=-1)

        # Generating the context from the attention weights and values 
        context = (attn_weights @ values).transpose(1,2).reshape(b, num_tokens, self.d_out)

        # Next cache will be used for the next forward pass.
        return self.out_proj(context), next_cache





