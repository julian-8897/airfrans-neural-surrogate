"""Transolver (Wu et al. 2024, ICML; arXiv:2402.02366) for unstructured meshes.

Reimplemented from the authors' AirfRANS code (github.com/thuml/Transolver,
MIT licence, Airfoil-Design-AirfRANS/models/Transolver.py), without the timm
and einops dependencies and without unused modules. Defaults follow the
authors' AirfRANS configuration.

Transolver is not a graph network: it ignores mesh connectivity and treats
the nodes as a point cloud. Each layer softly assigns nodes to a small number
of learned "slices", runs attention among the slice tokens, and broadcasts the
result back to the nodes, so cost is linear in the number of nodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from airfoil_surrogate.data import BOX


@dataclass(frozen=True)
class TransolverConfig:
    in_dim: int = 7
    out_dim: int = 4
    n_hidden: int = 256
    n_layers: int = 8
    n_head: int = 8
    mlp_ratio: int = 2
    slice_num: int = 32
    dropout: float = 0.0
    ref: int = 8  # reference grid is ref x ref points over the domain; 0 disables
    box: tuple[float, float, float, float] = BOX


class PhysicsAttention(nn.Module):
    """Slice, attend among slice tokens, de-slice (irregular-mesh variant)."""

    def __init__(self, dim: int, heads: int, slice_num: int, dropout: float):
        super().__init__()
        self.heads, self.dim_head = heads, dim // heads
        inner = self.dim_head * heads
        self.scale = self.dim_head**-0.5
        self.temperature = nn.Parameter(torch.full((1, heads, 1, 1), 0.5))
        self.in_project_x = nn.Linear(dim, inner)
        self.in_project_fx = nn.Linear(dim, inner)
        self.in_project_slice = nn.Linear(self.dim_head, slice_num)
        self.to_q = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_k = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_v = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def _heads(self, t: torch.Tensor) -> torch.Tensor:
        b, n, _ = t.shape
        return t.reshape(b, n, self.heads, self.dim_head).transpose(1, 2)  # B H N D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        fx = self._heads(self.in_project_fx(x))
        xs = self._heads(self.in_project_x(x))
        weights = torch.softmax(self.in_project_slice(xs) / self.temperature, dim=-1)
        norm = weights.sum(dim=2)  # B H G
        tokens = torch.einsum("bhnd,bhng->bhgd", fx, weights) / (norm[..., None] + 1e-5)

        q, k, v = self.to_q(tokens), self.to_k(tokens), self.to_v(tokens)
        attn = torch.softmax(q @ k.transpose(-1, -2) * self.scale, dim=-1)
        tokens = self.dropout(attn) @ v  # B H G D

        out = torch.einsum("bhgd,bhng->bhnd", tokens, weights)
        return self.to_out(out.transpose(1, 2).reshape(b, n, -1))


def _mlp(n_in: int, n_hidden: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_in, n_hidden), nn.GELU(), nn.Linear(n_hidden, n_out)
    )


class TransolverBlock(nn.Module):
    def __init__(self, cfg: TransolverConfig, last: bool):
        super().__init__()
        h = cfg.n_hidden
        self.ln_1 = nn.LayerNorm(h)
        self.attn = PhysicsAttention(h, cfg.n_head, cfg.slice_num, cfg.dropout)
        self.ln_2 = nn.LayerNorm(h)
        self.mlp = _mlp(h, h * cfg.mlp_ratio, h)
        self.head = (
            nn.Sequential(nn.LayerNorm(h), nn.Linear(h, cfg.out_dim)) if last else None
        )

    def forward(self, fx: torch.Tensor) -> torch.Tensor:
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        return self.head(fx) if self.head is not None else fx


class Transolver(nn.Module):
    """Map inputs (B, N, in_dim) and raw positions (B, N, 2) to (B, N, out_dim).

    With ``ref > 0`` the raw positions are expanded into distances to a fixed
    ref x ref grid over the domain (the authors' "unified position").
    """

    def __init__(self, cfg: TransolverConfig = TransolverConfig()):
        super().__init__()
        self.cfg = cfg
        n_extra = cfg.ref**2
        self.preprocess = _mlp(cfg.in_dim + n_extra, 2 * cfg.n_hidden, cfg.n_hidden)
        self.placeholder = nn.Parameter(torch.rand(cfg.n_hidden) / cfg.n_hidden)
        self.blocks = nn.ModuleList(
            TransolverBlock(cfg, last=(i == cfg.n_layers - 1))
            for i in range(cfg.n_layers)
        )
        if cfg.ref > 0:
            xmin, xmax, ymin, ymax = cfg.box
            gx, gy = torch.meshgrid(
                torch.linspace(xmin, xmax, cfg.ref),
                torch.linspace(ymin, ymax, cfg.ref),
                indexing="ij",
            )
            self.register_buffer("grid", torch.stack([gx, gy], -1).reshape(-1, 2))
        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.orthogonal_(block.attn.in_project_slice.weight)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        if self.cfg.ref > 0:
            x = torch.cat(
                [x, torch.cdist(pos, self.grid[None].expand(len(pos), -1, -1))], -1
            )
        fx = self.preprocess(x) + self.placeholder
        for block in self.blocks:
            fx = block(fx)
        return fx
