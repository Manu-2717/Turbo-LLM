import torch

class KVCache:
    def __init__(self):
        self.keys = {}
        self.values = {}

    def get(self, layer_id):
        """
        Returns (key, value) tensors for the given layer, or (None, None) if not present.
        """
        return self.keys.get(layer_id), self.values.get(layer_id)

    def append(self, layer_id, k, v):
        """
        Appends/concatenates new K and V states to the cache for the given layer.
        k, v: [batch_size, num_heads, seq_len, head_dim]
        """
        if layer_id not in self.keys or self.keys[layer_id] is None:
            self.keys[layer_id] = k.clone().detach() if hasattr(k, "clone") else k
            self.values[layer_id] = v.clone().detach() if hasattr(v, "clone") else v
        else:
            # Concatenate along the sequence dimension (dim=2)
            self.keys[layer_id] = torch.cat([self.keys[layer_id], k], dim=2)
            self.values[layer_id] = torch.cat([self.values[layer_id], v], dim=2)

    def clear(self):
        """
        Clears all cached keys and values.
        """
        self.keys.clear()
        self.values.clear()

    def update(self, key_states, value_states, layer_idx):
        """
        Compatibility method for Hugging Face Transformers attention layer forward pass.
        Appends new states and returns the full cached states.
        """
        self.append(layer_idx, key_states, value_states)
        return self.keys[layer_idx], self.values[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """
        Returns the sequence length of cached tokens for the given layer.
        """
        if len(self.keys) == 0 or layer_idx not in self.keys or self.keys[layer_idx] is None:
            return 0
        return self.keys[layer_idx].shape[-2]

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        """
        Returns a tuple (kv_length, kv_offset) corresponding to the length and offset 
        of the cache at the given layer index. Used to generate causal masks.
        """
        if layer_idx not in self.keys or self.keys[layer_idx] is None:
            return query_length, 0
        kv_length = self.get_seq_length(layer_idx) + query_length
        return kv_length, 0
