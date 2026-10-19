'''
    kv cache,
    Here we are designing things in very simple ways

'''
import torch

class KVCache:

    def __init__(self, n_layers):
        self.cache = [None] * n_layers
        
    def get(self, layer_idx):
        return self.cache[layer_idx]

    def update(self, layer_idx, cache):
        self.cache[layer_idx] = cache
        
    def get_all(self):
        return self.cache

    def reset(self):
        for i in range(len(self.cache)):
            self.cache[i] = None
    

