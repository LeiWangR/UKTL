"""Controlled KTL vs UKTL experiments on clean and nuisance-mode tasks.

Run from the repository root:
    python examples/toy_experiments.py
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from uktl import UKTLClassifier, UKTLFeatureExtractor
from examples.mode_noise_data import RANK, SHAPE, make_dataset, split_dataset

CLASS_COUNTS = (4, 10)
SEEDS = (0, 1, 2)
PIVOTS = 16
EPOCHS = 40
BANDWIDTH = 1.0
MU = 0.05
BETA = 0.01
SIGMA_LOW = 0.25
SIGMA_HIGH = 2.0
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0


@dataclass
class Result:
    val_accuracy: float
    test_accuracy: float
    clean_test_accuracy: float
    nuisance_test_accuracy: float
    sigma_means: List[float]


def build_model(seed: int, num_classes: int, uncertainty: bool) -> UKTLClassifier:
    """Build KTL/UKTL with identical non-MSN settings and seed-controlled init."""
    torch.manual_seed(seed)
    features = UKTLFeatureExtractor(
        tensor_shape=SHAPE,
        subspace_rank=RANK,
        num_pivots=PIVOTS,
        bandwidth=BANDWIDTH,
        mu_init=MU,
        learnable_mu=False,
        uncertainty=uncertainty,
        sigma_low=SIGMA_LOW,
        sigma_high=SIGMA_HIGH,
        nystrom_inverse_sqrt_method="newton_schulz",
        nystrom_inverse_sqrt_iterations=25,
    )
    torch.manual_seed(seed + 10000)
    return UKTLClassifier(features, num_classes)


def train_once(seed: int, num_classes: int, nuisance_mode: bool, uncertainty: bool) -> Result:
    torch.manual_seed(seed)
    x, y, corrupted = make_dataset(seed, num_classes, nuisance_mode)

    (
        train_x, train_y, _train_corrupted,
        val_x, val_y, _val_corrupted,
        test_x, test_y, test_corrupted,
    ) = split_dataset(x, y, corrupted, num_classes, seed)

    model = build_model(seed, num_classes, uncertainty)
    # Pivots are initialized only from the training split.
    model.features.initialize_pivots(train_x, soft_kmeans_iters=15, seed=seed)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_val = -1.0
    best_state = None

    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, stats = model(train_x, return_stats=True)
        loss = F.cross_entropy(logits, train_y)

        if uncertainty:
            # Full-batch optimization makes the normalization use all training
            # samples, rather than a changing minibatch subset.
            loss = loss + BETA * stats["sigma_regularizer"] / train_y.numel()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_acc = float((val_logits.argmax(dim=1) == val_y).float().mean())

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No best validation state was recorded.")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits, stats = model(test_x, return_stats=True)
        preds = test_logits.argmax(dim=1)
        test_acc = float((preds == test_y).float().mean())

        clean_mask = ~test_corrupted
        nuisance_mask = test_corrupted
        clean_acc = float((preds[clean_mask] == test_y[clean_mask]).float().mean())
        nuisance_acc = float(
            (preds[nuisance_mask] == test_y[nuisance_mask]).float().mean()
        ) if nuisance_mode else float("nan")

        sigma_means = [float(s.mean()) for s in stats["input_sigmas"]]

    return Result(best_val, test_acc, clean_acc, nuisance_acc, sigma_means)


def mean_std(values: List[float]) -> Tuple[float, float]:
    return mean(values), pstdev(values)


def summarize(task_name: str, num_classes: int, results: Dict[str, List[Result]]) -> None:
    print("\n" + "=" * 78)
    print(f"{task_name} | {num_classes} classes")
    print("=" * 78)

    for method, values in results.items():
        vm, vs = mean_std([r.val_accuracy for r in values])
        tm, ts = mean_std([r.test_accuracy for r in values])
        cm, cs = mean_std([r.clean_test_accuracy for r in values])

        print(f"\n{method}")
        print(f"  Val               : {100*vm:6.2f} ± {100*vs:5.2f}%")
        print(f"  Overall test      : {100*tm:6.2f} ± {100*ts:5.2f}%")
        print(f"  Clean-subset test : {100*cm:6.2f} ± {100*cs:5.2f}%")

        nuisance_values = [r.nuisance_test_accuracy for r in values]
        if all(torch.isfinite(torch.tensor(v)) for v in nuisance_values):
            nm, ns = mean_std(nuisance_values)
            print(
                f"  Nuisance-subset test: "
                f"{100*nm:6.2f} ± {100*ns:5.2f}%"
            )

        if method == "UKTL":
            mode_sigmas = [
                mean(r.sigma_means[m] for r in values)
                for m in range(len(values[0].sigma_means))
            ]
            print("  Mean sigma   : " + ", ".join(
                f"mode {m+1}={s:.3f}" for m, s in enumerate(mode_sigmas)
            ))


def run_task(num_classes: int, nuisance_mode: bool) -> Dict[str, List[Result]]:
    return {
        "KTL": [
            train_once(seed, num_classes, nuisance_mode, uncertainty=False)
            for seed in SEEDS
        ],
        "UKTL": [
            train_once(seed, num_classes, nuisance_mode, uncertainty=True)
            for seed in SEEDS
        ],
    }


def main() -> None:
    for num_classes in CLASS_COUNTS:
        for nuisance_mode, task_name in (
            (False, "Clean subspace task"),
            (True, "Nuisance-mode task"),
        ):
            results = run_task(num_classes, nuisance_mode)
            summarize(task_name, num_classes, results)

            ktl_test = mean(r.test_accuracy for r in results["KTL"])
            uktl_test = mean(r.test_accuracy for r in results["UKTL"])
            print(f"\n  UKTL - KTL test difference: {100*(uktl_test-ktl_test):+.2f} points")

            if nuisance_mode:
                ktl_n = mean(r.nuisance_test_accuracy for r in results["KTL"])
                uktl_n = mean(r.nuisance_test_accuracy for r in results["UKTL"])
                print(f"  UKTL - KTL nuisance difference: {100*(uktl_n-ktl_n):+.2f} points")


if __name__ == "__main__":
    main()
