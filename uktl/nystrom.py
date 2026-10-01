"""Differentiable Nyström kernel linearization and soft k-means pivots."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def soft_kmeans(
    x: Tensor,
    num_centers: int,
    num_iters: int = 20,
    temperature: float = 1.0,
    seed: int = 0,
) -> Tensor:
    """Soft k-means initializer for learnable Nyström pivots."""
    if x.ndim < 2:
        raise ValueError("x must have a batch dimension and at least one feature mode.")
    if not 1 <= num_centers <= x.shape[0]:
        raise ValueError("num_centers must be between 1 and the number of samples.")
    if num_iters < 1:
        raise ValueError("num_iters must be positive.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    flat = x.reshape(x.shape[0], -1)
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    init_idx = torch.randperm(flat.shape[0], generator=generator, device=x.device)[:num_centers]
    centers = flat[init_idx].clone()

    for _ in range(num_iters):
        dist2 = torch.cdist(flat, centers).square()
        weights = torch.softmax(-dist2 / temperature, dim=1)
        denom = weights.sum(dim=0).clamp_min(torch.finfo(flat.dtype).eps)
        centers = (weights.transpose(0, 1) @ flat) / denom[:, None]

    return centers.reshape(num_centers, *x.shape[1:])


def _symmetrize(matrix: Tensor, jitter: float) -> Tensor:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square 2-D.")
    n = matrix.shape[0]
    eye = torch.eye(n, dtype=matrix.dtype, device=matrix.device)
    return 0.5 * (matrix + matrix.transpose(-1, -2)) + jitter * eye


def inverse_sqrt_psd(
    matrix: Tensor,
    jitter: float = 1e-4,
    eps: float = 1e-6,
    method: str = "newton_schulz",
    iterations: int = 25,
) -> Tensor:
    """
    Symmetric inverse square root of a PSD/SPD matrix.

    ``method='eigh'`` follows the paper's Eq. (19) literally. ``method='newton_schulz'``
    computes the same matrix function with an iterative polynomial transform and avoids
    backpropagating through eigenvectors, which is typically more robust when learned
    Nyström pivots make the kernel matrix nearly degenerate.
    """
    if jitter < 0 or eps <= 0:
        raise ValueError("Require jitter >= 0 and eps > 0.")
    if method not in {"eigh", "newton_schulz"}:
        raise ValueError("method must be 'eigh' or 'newton_schulz'.")
    if iterations < 1:
        raise ValueError("iterations must be positive.")

    sym = _symmetrize(matrix, jitter)
    n = sym.shape[0]

    if method == "eigh":
        eigvals, eigvecs = torch.linalg.eigh(sym)
        safe = eigvals.clamp_min(eps)
        return eigvecs @ torch.diag(torch.rsqrt(safe)) @ eigvecs.transpose(-1, -2)

    # Newton-Schulz inverse square root. Normalize by trace so the spectrum lies
    # in (0, 1]; the added jitter ensures strict positivity.
    eye = torch.eye(n, dtype=sym.dtype, device=sym.device)
    scale = sym.diagonal().sum().clamp_min(eps)
    a = sym / scale
    y = a
    z = eye
    for _ in range(iterations):
        t = 0.5 * (3.0 * eye - z @ y)
        y = y @ t
        z = t @ z
    return z / torch.sqrt(scale)


class NystromLinearizer(nn.Module):
    """Turn a positive-definite tensor kernel into compact explicit features."""

    def __init__(
        self,
        num_pivots: int,
        jitter: float = 1e-4,
        eig_eps: float = 1e-6,
        inverse_sqrt_method: str = "newton_schulz",
        inverse_sqrt_iterations: int = 25,
    ) -> None:
        super().__init__()
        if num_pivots < 1:
            raise ValueError("num_pivots must be positive.")
        self.num_pivots = int(num_pivots)
        self.jitter = float(jitter)
        self.eig_eps = float(eig_eps)
        self.inverse_sqrt_method = inverse_sqrt_method
        self.inverse_sqrt_iterations = int(inverse_sqrt_iterations)

    def forward(self, k_xc: Tensor, k_cc: Tensor) -> Tensor:
        if k_xc.ndim != 2 or k_cc.ndim != 2:
            raise ValueError("Kernel matrices must be 2-D.")
        if k_xc.shape[-1] != self.num_pivots:
            raise ValueError("k_xc has the wrong number of pivots.")
        if k_cc.shape != (self.num_pivots, self.num_pivots):
            raise ValueError("k_cc has the wrong shape.")
        p_inv = inverse_sqrt_psd(
            k_cc,
            jitter=self.jitter,
            eps=self.eig_eps,
            method=self.inverse_sqrt_method,
            iterations=self.inverse_sqrt_iterations,
        )
        features = k_xc @ p_inv
        # Eq. (20): center each sample across the C pivot features.
        return features - features.mean(dim=-1, keepdim=True)
