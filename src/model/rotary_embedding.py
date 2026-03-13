# src/model/rotary_embedding.py

import math
import torch

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x.shape = (num_tokens, num_heads, head_dim)
    if x.dim() == 3:
        # cos, sin: (num_tokens, head_dim/2) -> (num_tokens, 1, head_dim/2)
        cos, sin = cos[:, None, :], sin[:, None, :]
        # (num_tokens, num_heads, head_dim) -> (num_tokens, num_heads, head_dim/2)
        x1, x2 = x.chunk(2, dim=-1)
        # broadcasting
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        # (num_tokens, num_heads, head_dim/2) -> (num_tokens, num_heads, head_dim)
        return torch.cat([y1, y2], dim=-1)
    # x.shape = (B, num_tokens, num_heads, head_dim)
    else:
        # cos, sin: (num_tokens, head_dim/2) -> (1, num_tokens, 1, head_dim/2)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        # (B, num_tokens, num_heads, head_dim) -> (B, num_tokens, num_heads, head_dim/2)
        x1, x2 = x.chunk(2, dim=-1)
        # broadcasting
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        # (B, num_tokens, num_heads, head_dim/2) -> (B, num_tokens, num_heads, head_dim)
        return torch.cat([y1, y2], dim=-1)

class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self, 
        base:int,
        rotary_embedding: int, 
        max_position: int = 2048,
        is_llama3: bool = False,
        # the following params are only used in llama3.2
        llama3_rope_factor: float = 32.0,
        llama3_rope_high_freq_factor: float = 4.0,
        llama3_rope_low_freq_factor: float = 1.0,
        llama3_rope_original_max_position_embeddings: int = 8192,
    ):
        super().__init__()
        self.base = base
        self.rotary_embedding = rotary_embedding
        self.max_position = max_position
        
        # (head_dim/2,) -> (head_dim/2,)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.rotary_embedding, 2).float() / self.rotary_embedding))
        if is_llama3:
            wave_len = 2 * math.pi / inv_freq
            if llama3_rope_low_freq_factor == llama3_rope_high_freq_factor:
                inv_freq = torch.where(
                    wave_len < llama3_rope_original_max_position_embeddings,
                    inv_freq,
                    inv_freq / llama3_rope_factor,
                )
            else:
                delta = llama3_rope_high_freq_factor - llama3_rope_low_freq_factor
                smooth = (llama3_rope_original_max_position_embeddings / wave_len - llama3_rope_low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / llama3_rope_factor + smooth
                inv_freq = factor * inv_freq
        self.inv_freq = inv_freq

        # (max_position,)
        positions = torch.arange(self.max_position).float()
        # (max_position,), (head_dim/2,) -> (max_position, head_dim/2)
        freqs = torch.einsum("i,j -> ij", positions, self.inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        # (max_position, head_dim/2) -> (max_position, head_dim)
        cos_sin_cache = torch.cat([cos, sin], dim=-1)
        self.register_buffer("cos_sin_cache", cos_sin_cache)
    
    @torch.compile
    def forward(self, positions, query, key):
        # positions: (num_tokens,)
        # query, key: (B, num_tokens, num_heads, head_dim) | (num_tokens, num_heads, head_dim)
        cos_sin_cache = self.cos_sin_cache[positions]   # type: ignore
        cos, sin = cos_sin_cache.chunk(2, dim=-1)
        query = apply_rotary_pos_emb(query, cos, sin)
        key = apply_rotary_pos_emb(key, cos, sin)
        return query, key

if __name__ == "__main__":
    # test
    base = 10000
    rotary_embedding = 64
    max_position = 2048
    model = RotaryEmbedding(base, rotary_embedding, max_position)
    positions = torch.arange(10)
    query = torch.randn(2, 10, 4, 64)
    key = torch.randn(2, 10, 4, 64)
    query, key = model(positions, query, key)
    print(query.shape) # (2, 10, 4, 64)
    print(key.shape) # (2, 10, 4, 64)