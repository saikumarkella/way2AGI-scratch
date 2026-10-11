# Key Value Cache for the Compute Optimization of the Attention Mechanism

'''
    KV
'''
import torch

class KVCache:

    # Here we are defining the Key, Value Cache for the Attention Mechanism.
    # Update will be done in the forward pass of the attention mechanism.
    # For training, we will be using the full sequence, so no need to use the cache.
    # For inference, we will be using the cache to store the previously computed the keys and values, To eliminate 
    # recomputation.

    def __init__(self, num_kv_groups, head_dim, max_seq_len):
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len 

        # Allocating the cache for the keys and values.
        self.keys_cache = torch.zeros((num_kv_groups, max_seq_len, head_dim))
        self.values_cache = torch.zeros((num_kv_groups, max_seq_len, head_dim))



    def update(self, new_keys, new_values, start_pos):
        # updating the cache with the new keys and values.
        self.keys_cache[:, start_pos:start_pos + new_keys.size(1), :] = new_keys
        self.values_cache[:, start_pos:start_pos + new_values.size(1), :] = new_values

    def get_cache(self):
        # returning the cache for the keys and values.
        return self.keys_cache, self.values_cache

