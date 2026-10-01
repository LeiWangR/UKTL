"""Small synthetic tensor dataset with class-specific Tucker subspaces."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def _orthonormal(rows: int, rank: int, generator: torch.Generator) -> Tensor:
    q, _ = torch.linalg.qr(torch.randn(rows, rank, generator=generator))
    return q[:, :rank]


def make_dataset(
    n_train: int = 96,
    n_test: int = 48,
    num_classes: int = 4,
    tensor_shape: Tuple[int, int, int] = (6, 7, 8),
    rank: int = 3,
    seed: int = 7,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generate tensors whose discriminative information lives in mode subspaces."""
    if rank > min(tensor_shape):
        raise ValueError("rank must not exceed the smallest tensor dimension.")

    g = torch.Generator().manual_seed(seed)
    class_bases = [
        [_orthonormal(dim, rank, g) for dim in tensor_shape]
        for _ in range(num_classes)
    ]

    def build(n: int) -> Tuple[Tensor, Tensor]:
        xs, ys = [], []
        for i in range(n):
            y = i % num_classes
            ys.append(y)
            factors = class_bases[y]
            # Shared low-rank core, with small sample-specific variation.
            core = torch.randn(rank, rank, rank, generator=g) * 0.6
            sample = torch.einsum(
                "abc,ia,jb,kc->ijk", core, factors[0], factors[1], factors[2]
            )
            # Add isotropic observation noise so subspace comparison matters.
            sample = sample + 0.12 * torch.randn(*tensor_shape, generator=g)
            xs.append(sample)
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

    return (*build(n_train), *build(n_test))
