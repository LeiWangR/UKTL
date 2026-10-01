"""Grassmann projection kernels used by Kernelized Tensor Learning."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


def squared_projection_distance(a: Tensor, b: Tensor) -> Tensor:
    """
    Pairwise Frobenius distance between AA^T and BB^T, squared.

    Parameters
    ----------
    a:
        [B, I, p] weighted bases.
    b:
        [C, I, p] weighted bases.

    Returns
    -------
    Tensor
        [B, C] matrix with ||AA^T - BB^T||_F^2.

    This avoids explicitly materializing I x I projection matrices. When the
    bases are orthonormal, the expression reduces to 2p - 2||A^T B||_F^2.
    """
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("a and b must have shape [N, I, p].")
    if a.shape[1] != b.shape[1] or a.shape[2] != b.shape[2]:
        raise ValueError(
            f"Incompatible subspaces: {tuple(a.shape)} vs {tuple(b.shape)}."
        )

    # Gram(A) and Gram(B): tr((AA^T)^2) = ||A^T A||_F^2.
    gram_a = torch.einsum("bip,biq->bpq", a, a)
    gram_b = torch.einsum("cip,ciq->cpq", b, b)
    self_a = gram_a.square().sum(dim=(-2, -1))
    self_b = gram_b.square().sum(dim=(-2, -1))

    # Cross(A,B) = A^T B, pairwise over batch and pivots.
    cross = torch.einsum("bip,ciq->bcpq", a, b)
    cross_sq = cross.square().sum(dim=(-2, -1))

    d2 = self_a[:, None] + self_b[None, :] - 2.0 * cross_sq
    return d2.clamp_min(0.0)


def grassmann_rbf_kernel(a: Tensor, b: Tensor, bandwidth: float) -> Tensor:
    """RBF kernel over projection-matrix embeddings of subspaces."""
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")
    d2 = squared_projection_distance(a, b)
    return torch.exp(-d2 / (2.0 * bandwidth * bandwidth))


def sum_product_tensor_kernel(
    a_modes: Sequence[Tensor],
    b_modes: Sequence[Tensor],
    bandwidth: float,
    mu: Tensor | float = 0.5,
) -> Tensor:
    """
    UKTL/KTL sum-product tensor kernel.

    k(X,Z) = mu * sum_m k_m(X_m,Z_m) + (1-mu) * prod_m k_m(X_m,Z_m).
    """
    if len(a_modes) == 0 or len(a_modes) != len(b_modes):
        raise ValueError("a_modes and b_modes must contain the same non-empty modes.")

    factors = [
        grassmann_rbf_kernel(a, b, bandwidth)
        for a, b in zip(a_modes, b_modes)
    ]
    stacked = torch.stack(factors, dim=0)
    sum_part = stacked.sum(dim=0)
    product_part = stacked.prod(dim=0)

    mu_t = torch.as_tensor(mu, dtype=sum_part.dtype, device=sum_part.device)
    mu_t = mu_t.clamp(0.0, 1.0)
    return mu_t * sum_part + (1.0 - mu_t) * product_part
