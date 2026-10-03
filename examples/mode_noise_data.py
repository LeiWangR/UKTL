"""Controlled synthetic data for the UKTL unequal-mode-reliability benchmark."""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

SHAPE = (10, 10, 10)
RANK = 2
TOTAL_SAMPLES = 400
CORRUPTION_RATE = 0.50
SIGNAL_STRENGTH = 1.0
NUISANCE_STRENGTH = 2.0
OBSERVATION_NOISE = 0.01


def class_bases(num_classes: int, dim: int, rank: int) -> list[Tensor]:
    """Construct class-dependent rank-2 subspaces.

    Each class contains one class-specific coordinate direction and a second
    direction orthogonal to a shared reference vector. The resulting class
    subspaces retain a common shared component while providing class-specific
    variation.
    """
    if num_classes not in (4, 10):
        raise ValueError("Supports 4 or 10 classes.")
    if rank != 2 or num_classes > dim:
        raise ValueError("This benchmark expects rank=2 and num_classes <= dim.")

    shared = torch.ones(dim, 1)
    shared = shared / shared.norm()
    bases = []
    for c in range(num_classes):
        e = torch.zeros(dim, 1)
        e[c, 0] = 1.0
        v = e - (e.T @ shared) * shared
        v = v / v.norm()
        bases.append(torch.cat([e, v], dim=1))
    return bases


def make_dataset(
    seed: int,
    num_classes: int,
    nuisance_mode: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Generate paired clean/nuisance tensor data.

    The clean and nuisance datasets use the same class structure, signal, and
    observation noise for a given seed. The nuisance condition differs only by
    adding a strong, sample-dependent component with variation across mode 3
    for exactly 50% of samples. The nuisance is generated from the same
    mode-1/2 class factors, so those two modes remain class-specific.
    """
    if num_classes not in (4, 10):
        raise ValueError("This controlled benchmark supports 4 or 10 classes.")

    g = torch.Generator().manual_seed(seed)
    bases = [class_bases(num_classes, d, RANK) for d in SHAPE]

    xs, ys, corrupted = [], [], []
    per_class = TOTAL_SAMPLES // num_classes

    for i in range(TOTAL_SAMPLES):
        label = i % num_classes
        class_sample = i // num_classes
        f1, f2, f3 = (bases[m][label] for m in range(3))

        core = SIGNAL_STRENGTH * torch.randn(RANK, RANK, RANK, generator=g)
        signal = torch.einsum("abc,ia,jb,kc->ijk", core, f1, f2, f3)

        # Deterministic 50/50 nuisance assignment within every class.
        bad = bool(nuisance_mode and (class_sample < per_class * CORRUPTION_RATE))

        # Generate the nuisance for every sample so clean and nuisance datasets
        # consume the same random stream and differ only by its addition.
        nuisance_core = NUISANCE_STRENGTH * torch.randn(
            RANK, RANK, SHAPE[2], generator=g
        )
        nuisance = torch.einsum("abk,ia,jb->ijk", nuisance_core, f1, f2)

        sample = signal + nuisance if bad else signal
        sample = sample + OBSERVATION_NOISE * torch.randn(*SHAPE, generator=g)

        xs.append(sample)
        ys.append(label)
        corrupted.append(bad)

    return (
        torch.stack(xs),
        torch.tensor(ys, dtype=torch.long),
        torch.tensor(corrupted, dtype=torch.bool),
    )


def split_dataset(
    x: Tensor,
    y: Tensor,
    corrupted: Tensor,
    num_classes: int,
    seed: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Create class-balanced and corruption-balanced train/val/test splits.

    Clean condition: 50/20/30% of each class.
    Nuisance condition: 50/20/30% within each class × corruption group.
    Therefore no split choice depends on model outputs or test performance.
    """
    if len(x) != TOTAL_SAMPLES:
        raise ValueError("Unexpected sample count.")

    per_class = TOTAL_SAMPLES // num_classes
    n_train = per_class // 2
    n_val = per_class // 5
    g = torch.Generator().manual_seed(1000 + seed)

    train_idx, val_idx, test_idx = [], [], []

    if bool(corrupted.any()):
        # Nuisance task: split clean and corrupted members separately within
        # each class, preserving exactly the 50% nuisance proportion.
        for cls in range(num_classes):
            for is_bad in (False, True):
                idx = torch.nonzero(
                    (y == cls) & (corrupted == is_bad), as_tuple=False
                ).flatten()
                idx = idx[torch.randperm(idx.numel(), generator=g)]
                group_n = idx.numel()
                group_train = group_n // 2
                group_val = group_n // 5
                train_idx.append(idx[:group_train])
                val_idx.append(idx[group_train:group_train + group_val])
                test_idx.append(idx[group_train + group_val:])
    else:
        # Clean task: split each class into 50/20/30%.
        for cls in range(num_classes):
            idx = torch.nonzero(y == cls, as_tuple=False).flatten()
            idx = idx[torch.randperm(idx.numel(), generator=g)]
            train_idx.append(idx[:n_train])
            val_idx.append(idx[n_train:n_train + n_val])
            test_idx.append(idx[n_train + n_val:])

    train_idx = torch.cat(train_idx)
    val_idx = torch.cat(val_idx)
    test_idx = torch.cat(test_idx)

    train_idx = train_idx[torch.randperm(train_idx.numel(), generator=g)]
    val_idx = val_idx[torch.randperm(val_idx.numel(), generator=g)]
    test_idx = test_idx[torch.randperm(test_idx.numel(), generator=g)]

    return (
        x[train_idx], y[train_idx], corrupted[train_idx],
        x[val_idx], y[val_idx], corrupted[val_idx],
        x[test_idx], y[test_idx], corrupted[test_idx],
    )
