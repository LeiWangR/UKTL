#!/usr/bin/env python3
"""Train the task-agnostic UKTL module on a .pt dataset or built-in demo."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor

from examples.synthetic_data import make_dataset
from uktl import UKTLClassifier, UKTLFeatureExtractor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(path: str | None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    if path is None:
        return make_dataset()

    obj = torch.load(path, map_location="cpu", weights_only=False)
    required = ("train_x", "train_y", "test_x", "test_y")
    if not isinstance(obj, dict) or any(k not in obj for k in required):
        raise ValueError(
            "Dataset .pt file must be a dict containing train_x, train_y, test_x, test_y."
        )
    return tuple(obj[k].float() if "x" in k else obj[k].long() for k in required)  # type: ignore[return-value]


def accuracy(logits: Tensor, y: Tensor) -> float:
    return float((logits.argmax(dim=1) == y).float().mean().item())


def split_train_validation(
    x: Tensor,
    y: Tensor,
    val_fraction: float,
    seed: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1).")
    n = x.shape[0]
    n_val = max(1, int(round(n * val_fraction)))
    if n - n_val < 1:
        raise ValueError("Not enough training samples after validation split.")
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    perm = torch.randperm(n, generator=generator, device=x.device)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def build_model(tensor_shape: Tuple[int, ...], num_classes: int, args: argparse.Namespace) -> UKTLClassifier:
    features = UKTLFeatureExtractor(
        tensor_shape=tensor_shape,
        subspace_rank=args.rank,
        num_pivots=args.pivots,
        bandwidth=args.bandwidth,
        mu_init=args.mu,
        learnable_mu=args.learnable_mu,
        uncertainty=not args.no_uncertainty,
        sigma_low=args.sigma_low,
        sigma_high=args.sigma_high,
        nystrom_inverse_sqrt_method=args.nystrom_method,
        nystrom_inverse_sqrt_iterations=args.nystrom_iterations,
    )
    return UKTLClassifier(features, num_classes)


def main() -> None:
    p = argparse.ArgumentParser(description="Train UKTL on generic tensor data.")
    p.add_argument("--data", type=str, default=None, help="Optional .pt dataset file.")
    p.add_argument("--output", type=str, default="runs/demo", help="Output directory.")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=3, help="Subspace dimension p.")
    p.add_argument("--pivots", type=int, default=12, help="Number of Nyström pivots C.")
    p.add_argument("--bandwidth", type=float, default=1.0)
    p.add_argument("--mu", type=float, default=0.5, help="Initial sum/product mixture weight.")
    p.add_argument("--learnable-mu", action="store_true",
                    help="Optimize mu jointly. The paper tunes mu as a hyperparameter; this is an optional extension.")
    p.add_argument("--no-uncertainty", action="store_true")
    p.add_argument("--sigma-low", type=float, default=0.25,
                    help="Implementation choice: lower sigma bound for bounded MSN output.")
    p.add_argument("--sigma-high", type=float, default=2.0,
                    help="Implementation choice: upper sigma bound for bounded MSN output.")
    p.add_argument("--beta", type=float, default=0.01, help="Uncertainty regularization weight.")
    p.add_argument("--val-fraction", type=float, default=0.10,
                    help="Fraction of train_x used for checkpoint selection (never test_x).")
    p.add_argument("--nystrom-method", choices=["newton_schulz", "eigh"], default="newton_schulz",
                    help="Inverse-square-root implementation. 'eigh' follows Eq. (19) literally; the default avoids eigenvector backpropagation.")
    p.add_argument("--nystrom-iterations", type=int, default=25)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    train_x, train_y, test_x, test_y = load_data(args.data)
    if train_x.ndim < 3:
        raise ValueError("train_x must have shape [N, I1, ..., IM].")
    if train_x.shape[0] != train_y.shape[0]:
        raise ValueError("train_x and train_y must contain the same number of samples.")
    if test_x.shape[0] != test_y.shape[0]:
        raise ValueError("test_x and test_y must contain the same number of samples.")

    # The test set is held out completely from model selection.
    train_x, train_y, val_x, val_y = split_train_validation(
        train_x.float(), train_y.long(), args.val_fraction, args.seed
    )
    train_x, train_y = train_x.to(device), train_y.to(device)
    val_x, val_y = val_x.to(device), val_y.to(device)
    # Keep the held-out test set off-device and untouched until model selection is complete.

    tensor_shape = tuple(train_x.shape[1:])
    # Determine the classifier size from training/validation labels only.
    # The held-out test labels are intentionally not consulted until final evaluation.
    num_classes = int(torch.max(torch.cat([train_y, val_y])).item()) + 1
    model = build_model(tensor_shape, num_classes, args).to(device)
    model.features.initialize_pivots(
        train_x,
        soft_kmeans_iters=15,
        soft_kmeans_temperature=1.0,
        seed=args.seed,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, stats = model(train_x, return_stats=True)
        ce = criterion(logits, train_y)
        # Eq. (22) is written as a sum over samples. Divide both the CE and
        # regularizer by N so the optimization objective has an invariant scale.
        reg = stats["sigma_regularizer"] if not args.no_uncertainty else train_x.new_zeros(())
        reg = reg / train_y.numel()
        loss = ce + args.beta * reg
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at epoch {epoch}: ce={ce.item()}, reg={float(reg)}"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_logits = model(train_x)
            val_logits = model(val_x)
        train_acc = accuracy(train_logits, train_y)
        val_acc = accuracy(val_logits, val_y)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:03d} | loss {loss.item():.4f} | "
                f"train {train_acc*100:.1f}% | val {val_acc*100:.1f}% | "
                f"mu {float(model.features.mu.detach()):.3f}"
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    test_x, test_y = test_x.to(device), test_y.to(device)
    with torch.no_grad():
        test_logits = model(test_x)
    test_acc = accuracy(test_logits, test_y)

    checkpoint: Dict[str, object] = {
        "state_dict": best_state,
        "config": {
            "tensor_shape": tensor_shape,
            "num_classes": num_classes,
            "rank": args.rank,
            "pivots": args.pivots,
            "bandwidth": args.bandwidth,
            "mu": args.mu,
            "learnable_mu": args.learnable_mu,
            "no_uncertainty": args.no_uncertainty,
            "sigma_low": args.sigma_low,
            "sigma_high": args.sigma_high,
            "nystrom_method": args.nystrom_method,
            "nystrom_iterations": args.nystrom_iterations,
        },
        "best_validation_accuracy": best_val,
        "final_test_accuracy": test_acc,
    }
    ckpt_path = out_dir / "best.pt"
    torch.save(checkpoint, ckpt_path)
    (out_dir / "config.json").write_text(json.dumps(checkpoint["config"], indent=2, default=str))
    print(f"saved: {ckpt_path}")
    print(f"best validation accuracy: {best_val*100:.2f}%")
    print(f"final held-out test accuracy: {test_acc*100:.2f}%")


if __name__ == "__main__":
    main()
