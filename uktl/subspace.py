"""Tensor unfolding and differentiable mode-wise subspace extraction."""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch import Tensor


def validate_tensor_batch(x: Tensor) -> None:
    """Validate a batched tensor of shape [B, I1, ..., IM]."""
    if x.ndim < 3:
        raise ValueError(
            "Expected a batched tensor with shape [B, I1, ..., IM] and at least "
            "2 tensor modes."
        )
    if x.shape[0] < 1:
        raise ValueError("The batch dimension must be non-empty.")


def mode_unfold(x: Tensor, mode: int) -> Tensor:
    """
    Compute a batched mode-m unfolding.

    Parameters
    ----------
    x:
        Tensor with shape [B, I1, ..., IM].
    mode:
        Zero-based tensor mode in [0, M-1].

    Returns
    -------
    Tensor
        Shape [B, I_mode, prod(I_k for k != mode)].
    """
    validate_tensor_batch(x)
    num_modes = x.ndim - 1
    if not 0 <= mode < num_modes:
        raise ValueError(f"mode must be in [0, {num_modes - 1}], got {mode}.")

    # Keep batch first; move the selected mode next; flatten the remaining modes.
    perm = [0, mode + 1] + [i for i in range(1, x.ndim) if i != mode + 1]
    y = x.permute(perm).contiguous()
    return y.reshape(x.shape[0], x.shape[mode + 1], -1)


def extract_mode_subspaces(x: Tensor, rank: int) -> List[Tensor]:
    """
    Extract top-rank left singular-vector bases for every tensor mode.

    The operation uses differentiable SVD in ordinary non-degenerate cases.
    Singular-vector gradients can become unstable or undefined at repeated or
    nearly repeated singular values. For x shaped [B, I1, ..., IM], each output
    has shape [B, Im, rank].
    """
    validate_tensor_batch(x)
    if not isinstance(rank, int) or rank < 1:
        raise ValueError("rank must be a positive integer.")

    subspaces: List[Tensor] = []
    for mode in range(x.ndim - 1):
        unfolding = mode_unfold(x, mode)
        max_rank = min(unfolding.shape[-2], unfolding.shape[-1])
        if rank > max_rank:
            raise ValueError(
                f"rank={rank} is too large for mode {mode}: "
                f"maximum possible rank is {max_rank}."
            )
        u, _, _ = torch.linalg.svd(unfolding, full_matrices=False)
        subspaces.append(u[..., :rank])
    return subspaces


def projection_matrices(subspaces: Sequence[Tensor]) -> List[Tensor]:
    """Convert mode-wise bases U [B, I, p] to projection matrices UU^T."""
    return [u @ u.transpose(-1, -2) for u in subspaces]


def tensor_mode_shapes(x: Tensor) -> List[int]:
    """Return tensor mode sizes, excluding the batch dimension."""
    validate_tensor_batch(x)
    return list(x.shape[1:])
