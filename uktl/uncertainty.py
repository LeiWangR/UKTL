"""Multi-mode uncertainty estimation for UKTL."""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch import Tensor, nn


class ScaledSigmoid(nn.Module):
    """Map unconstrained values to a configurable positive bounded interval."""

    def __init__(self, low: float = 0.25, high: float = 2.0) -> None:
        super().__init__()
        if low <= 0 or high <= low:
            raise ValueError("Require 0 < low < high for ScaledSigmoid.")
        self.low = float(low)
        self.high = float(high)

    def forward(self, x: Tensor) -> Tensor:
        return self.low + (self.high - self.low) * torch.sigmoid(x)


class MultiModeSigmaNet(nn.Module):
    """
    Mode-specific uncertainty network.

    Each branch consumes the flattened projection matrix U U^T for one tensor
    mode and predicts p positive bounded variance-like values sigma. The UKTL
    confidence gate is 1/sqrt(sigma), applied per SVD direction (column of U).

    The paper's appendix clarifies that these values correspond to the p latent
    subspace directions. This implementation therefore scales U column-wise.
    """

    def __init__(
        self,
        mode_dims: Sequence[int],
        subspace_rank: int,
        sigma_low: float = 0.25,
        sigma_high: float = 2.0,
    ) -> None:
        super().__init__()
        if len(mode_dims) == 0:
            raise ValueError("mode_dims must contain at least one mode size.")
        if subspace_rank < 1:
            raise ValueError("subspace_rank must be positive.")

        self.mode_dims = tuple(int(d) for d in mode_dims)
        self.subspace_rank = int(subspace_rank)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d * d, self.subspace_rank),
                    ScaledSigmoid(sigma_low, sigma_high),
                )
                for d in self.mode_dims
            ]
        )

    def forward(self, subspaces: Sequence[Tensor]) -> List[Tensor]:
        if len(subspaces) != len(self.branches):
            raise ValueError(
                f"Expected {len(self.branches)} mode subspaces, got {len(subspaces)}."
            )

        sigmas: List[Tensor] = []
        for branch, u in zip(self.branches, subspaces):
            if u.ndim != 3:
                raise ValueError("Each subspace must have shape [B, I, p].")
            if u.shape[-1] != self.subspace_rank:
                raise ValueError(
                    f"Expected subspace rank {self.subspace_rank}, got {u.shape[-1]}."
                )
            proj = u @ u.transpose(-1, -2)
            sigmas.append(branch(proj.flatten(start_dim=-2)))
        return sigmas


def confidence_gate(sigmas: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Convert positive sigma values to direction-wise confidence weights.

    For sigma shaped [B, p], returns the same shape. This is the direct
    implementation of u = 1/sqrt(sigma) described in the paper appendix.
    """
    if torch.any(sigmas <= 0):
        raise ValueError("Sigma values must be positive.")
    return torch.rsqrt(sigmas.clamp_min(eps))


def apply_confidence_gate(subspaces: Sequence[Tensor], sigmas: Sequence[Tensor]) -> List[Tensor]:
    """Scale each subspace basis column by its predicted confidence."""
    if len(subspaces) != len(sigmas):
        raise ValueError("subspaces and sigmas must have the same number of modes.")
    return [u * confidence_gate(s).unsqueeze(-2) for u, s in zip(subspaces, sigmas)]


def uncertainty_regularizer(
    sigmas: Sequence[Tensor],
    eps: float = 0.0,
    reduction: str = "sum",
) -> Tensor:
    """Eq. (22) uncertainty term, with optional batch-size normalization.

    ``reduction='sum'`` is the literal summation written in the paper.
    ``reduction='mean'`` divides that value by the number of scalar sigma
    entries and is provided only as a convenience for demos.
    """
    if len(sigmas) == 0:
        raise ValueError("sigmas cannot be empty.")
    if reduction not in {"sum", "mean"}:
        raise ValueError("reduction must be 'sum' or 'mean'.")

    total = sigmas[0].new_zeros(())
    count = 0
    for sigma in sigmas:
        if sigma.ndim != 2:
            raise ValueError("Each sigma tensor must have shape [B, p].")
        denom = sigma.sum(dim=0, keepdim=True) + 1.0
        total = total + torch.log((sigma + 1.0 + eps) / denom).sum()
        count += sigma.numel()

    if reduction == "mean":
        return total / max(count, 1)
    return total
