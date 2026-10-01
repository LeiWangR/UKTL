"""Controlled toy experiments comparing KTL and UKTL.

The experiments are small and task-agnostic. They demonstrate the
intended behavior of UKTL on tensors whose class information is encoded in
mode-wise subspaces, with an additional nuisance-mode setting where one mode is
class-independent and varies strongly from sample to sample.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Dict, List, Tuple

import torch

from uktl import UKTLClassifier, UKTLFeatureExtractor


NUM_CLASSES = 4
SHAPE = (5, 6, 7)
RANK = 2
PIVOTS = 10
EPOCHS = 35
SEEDS = (0, 1, 2)


@dataclass
class Result:
    val_accuracy: float
    test_accuracy: float
    sigma_means: List[float]


def orthonormal(rows: int, rank: int, generator: torch.Generator) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(rows, rank, generator=generator), mode="reduced")
    return q[:, :rank]


def make_dataset(seed: int, nuisance_mode: bool) -> Tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    class_bases = [
        [orthonormal(d, RANK, generator) for d in SHAPE]
        for _ in range(NUM_CLASSES)
    ]

    def build(num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        xs: List[torch.Tensor] = []
        ys: List[int] = []
        for i in range(num_samples):
            label = i % NUM_CLASSES
            factors = class_bases[label]
            core = 0.8 * torch.randn(RANK, RANK, RANK, generator=generator)
            # In the nuisance setting, mode 3 is random and contains no class
            # information; modes 1 and 2 retain the discriminative subspaces.
            mode3 = orthonormal(SHAPE[2], RANK, generator) if nuisance_mode else factors[2]
            sample = torch.einsum(
                "abc,ia,jb,kc->ijk", core, factors[0], factors[1], mode3
            )
            sample = sample + 0.08 * torch.randn(*SHAPE, generator=generator)
            xs.append(sample)
            ys.append(label)
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

    train_x, train_y = build(160)
    test_x, test_y = build(160)
    return train_x, train_y, test_x, test_y


def split_train_validation(
    x: torch.Tensor, y: torch.Tensor, seed: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1000 + seed)
    perm = torch.randperm(len(x), generator=generator)
    val_idx = perm[:32]
    train_idx = perm[32:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def run_once(seed: int, nuisance_mode: bool, uncertainty: bool) -> Result:
    torch.manual_seed(seed)
    train_x, train_y, test_x, test_y = make_dataset(seed, nuisance_mode)
    train_x, train_y, val_x, val_y = split_train_validation(train_x, train_y, seed)

    features = UKTLFeatureExtractor(
        tensor_shape=SHAPE,
        subspace_rank=RANK,
        num_pivots=PIVOTS,
        bandwidth=1.0,
        mu_init=0.5,
        learnable_mu=False,
        uncertainty=uncertainty,
        sigma_low=0.25,
        sigma_high=2.0,
        nystrom_inverse_sqrt_method="newton_schulz",
    )
    model = UKTLClassifier(features, NUM_CLASSES)
    features.initialize_pivots(train_x, soft_kmeans_iters=10, seed=seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    best_val = -1.0
    best_state = None

    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, stats = model(train_x, return_stats=True)
        loss = torch.nn.functional.cross_entropy(logits, train_y)
        if uncertainty:
            loss = loss + 0.01 * stats["sigma_regularizer"] / train_y.numel()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_acc = float((model(val_x).argmax(1) == val_y).float().mean())
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_acc = float((model(test_x).argmax(1) == test_y).float().mean())
        stats = model(test_x, return_stats=True)[1]
        sigmas = [float(s.mean()) for s in stats["input_sigmas"]]
    return Result(best_val, test_acc, sigmas)


def summarize(results: Dict[str, List[Result]]) -> None:
    for label, values in results.items():
        test_scores = [r.test_accuracy for r in values]
        print(
            f"{label:<12} test = {100 * mean(test_scores):5.2f} ± "
            f"{100 * pstdev(test_scores):4.2f}%"
        )
        val_scores = [r.val_accuracy for r in values]
        print(
            f"{'':<12} val  = {100 * mean(val_scores):5.2f} ± "
            f"{100 * pstdev(val_scores):4.2f}%"
        )
        if values[0].sigma_means:
            mode_means = [mean(r.sigma_means[m] for r in values) for m in range(len(values[0].sigma_means))]
            print(f"{'':<12} mean sigma = " + ", ".join(f"{v:.3f}" for v in mode_means))


def main() -> None:
    for nuisance_mode, title in [
        (False, "Clean subspace task"),
        (True, "Nuisance-mode task"),
    ]:
        print(f"\n{title}")
        results = {
            "KTL": [run_once(seed, nuisance_mode, uncertainty=False) for seed in SEEDS],
            "UKTL": [run_once(seed, nuisance_mode, uncertainty=True) for seed in SEEDS],
        }
        summarize(results)
        if nuisance_mode:
            print("Higher sigma in the nuisance mode is an intended qualitative diagnostic, not a target value.")


if __name__ == "__main__":
    main()
