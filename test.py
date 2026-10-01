#!/usr/bin/env python3
"""Evaluate a UKTL checkpoint on a .pt dataset or the built-in demo."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor

from examples.synthetic_data import make_dataset
from uktl import UKTLClassifier, UKTLFeatureExtractor


def load_data(path: str | None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    if path is None:
        return make_dataset()
    obj = torch.load(path, map_location="cpu", weights_only=False)
    required = ("train_x", "train_y", "test_x", "test_y")
    if not isinstance(obj, dict) or any(k not in obj for k in required):
        raise ValueError("Dataset .pt file must contain train_x, train_y, test_x, test_y.")
    return tuple(obj[k].float() if "x" in k else obj[k].long() for k in required)  # type: ignore[return-value]


def main() -> None:
    p = argparse.ArgumentParser(description="Test a trained UKTL model.")
    p.add_argument("--checkpoint", type=str, default="runs/demo/best.pt")
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    train_x, train_y, test_x, test_y = load_data(args.data)
    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = checkpoint["config"]
    features = UKTLFeatureExtractor(
        tensor_shape=tuple(cfg["tensor_shape"]),
        subspace_rank=int(cfg["rank"]),
        num_pivots=int(cfg["pivots"]),
        bandwidth=float(cfg["bandwidth"]),
        mu_init=float(cfg["mu"]),
        learnable_mu=bool(cfg.get("learnable_mu", False)),
        uncertainty=not bool(cfg["no_uncertainty"]),
        sigma_low=float(cfg["sigma_low"]),
        sigma_high=float(cfg["sigma_high"]),
        nystrom_inverse_sqrt_method=str(cfg.get("nystrom_method", "eigh")),
        nystrom_inverse_sqrt_iterations=int(cfg.get("nystrom_iterations", 25)),
    )
    model = UKTLClassifier(features, int(cfg["num_classes"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        logits, stats = model(test_x, return_stats=True)
        preds = logits.argmax(dim=1)
        acc = (preds == test_y).float().mean().item()

    print(f"test accuracy: {acc*100:.2f}%")
    print(f"mixture mu: {float(stats['mu']):.4f}")
    if stats["input_sigmas"]:
        means = [float(s.mean()) for s in stats["input_sigmas"]]
        print("mean sigma per mode:", ", ".join(f"{v:.4f}" for v in means))

    # Save predictions next to the checkpoint for reproducibility.
    pred_path = Path(args.checkpoint).with_name("predictions.pt")
    torch.save({"predictions": preds.cpu(), "labels": test_y.cpu()}, pred_path)
    print(f"saved: {pred_path}")


if __name__ == "__main__":
    main()
