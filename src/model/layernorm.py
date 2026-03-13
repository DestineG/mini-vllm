# src/model/layrernorm.py

import torch

class LayerNorm(torch.nn.Module):
    def __init__(self, gamma: torch.Tensor, eps: float = 1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(gamma.detach().clone())
        self.eps = eps

    @property
    def gamma(self):
        return self.weight

    @torch.compile
    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            x = x + residual
        _rms_inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # 计算 1/RMS
        out = x * _rms_inv * self.weight
        return out, x
