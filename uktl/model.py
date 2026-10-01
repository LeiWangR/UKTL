"""Task-agnostic UKTL feature extractor and classifier."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .kernel import sum_product_tensor_kernel
from .nystrom import NystromLinearizer, soft_kmeans
from .subspace import extract_mode_subspaces
from .uncertainty import MultiModeSigmaNet, apply_confidence_gate, uncertainty_regularizer


class UKTLFeatureExtractor(nn.Module):
    """
    Core Uncertainty-driven Kernel Tensor Learning module.

    Input:  x shaped [B, I1, ..., IM].
    Output: explicit Nyström features shaped [B, C].

    No action-recognition-specific encoder is included. Users can feed raw
    tensors or the output of any domain-specific tensor encoder.
    """

    def __init__(
        self,
        tensor_shape: Sequence[int],
        subspace_rank: int = 4,
        num_pivots: int = 16,
        bandwidth: float = 1.0,
        mu_init: float = 0.5,
        learnable_mu: bool = False,
        uncertainty: bool = True,
        sigma_low: float = 0.25,
        sigma_high: float = 2.0,
        nystrom_jitter: float = 1e-4,
        nystrom_eps: float = 1e-6,
        nystrom_inverse_sqrt_method: str = "newton_schulz",
        nystrom_inverse_sqrt_iterations: int = 25,
    ) -> None:
        super().__init__()
        self.tensor_shape = tuple(int(s) for s in tensor_shape)
        if len(self.tensor_shape) < 2:
            raise ValueError("tensor_shape must describe at least two tensor modes.")
        if any(s < 1 for s in self.tensor_shape):
            raise ValueError("All tensor dimensions must be positive.")
        if subspace_rank < 1:
            raise ValueError("subspace_rank must be positive.")
        if num_pivots < 1:
            raise ValueError("num_pivots must be positive.")
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive.")

        self.subspace_rank = int(subspace_rank)
        self.num_pivots = int(num_pivots)
        self.bandwidth = float(bandwidth)
        self.use_uncertainty = bool(uncertainty)

        mu_init = float(mu_init)
        if not 0.0 <= mu_init <= 1.0:
            raise ValueError("mu_init must lie in [0, 1].")
        # Parameterize mu in unconstrained space to guarantee [0,1].
        if learnable_mu:
            init = torch.logit(torch.tensor(mu_init).clamp(1e-4, 1 - 1e-4))
            self.raw_mu = nn.Parameter(init)
        else:
            self.register_buffer("fixed_mu", torch.tensor(mu_init))
            self.raw_mu = None

        # Pivots always exist in the state dict; values are initialized from training
        # data before the first forward pass.
        self.pivots = nn.Parameter(torch.zeros(self.num_pivots, *self.tensor_shape))
        self.register_buffer("_initialized", torch.tensor(False), persistent=True)

        self.msn: Optional[MultiModeSigmaNet]
        if self.use_uncertainty:
            self.msn = MultiModeSigmaNet(
                mode_dims=self.tensor_shape,
                subspace_rank=self.subspace_rank,
                sigma_low=sigma_low,
                sigma_high=sigma_high,
            )
        else:
            self.msn = None

        self.nystrom = NystromLinearizer(
            num_pivots=num_pivots,
            jitter=nystrom_jitter,
            eig_eps=nystrom_eps,
            inverse_sqrt_method=nystrom_inverse_sqrt_method,
            inverse_sqrt_iterations=nystrom_inverse_sqrt_iterations,
        )

    @property
    def mu(self) -> Tensor:
        if self.raw_mu is None:
            return self.fixed_mu
        return torch.sigmoid(self.raw_mu)

    @property
    def initialized(self) -> bool:
        return bool(self._initialized.item())

    @torch.no_grad()
    def initialize_pivots(
        self,
        x: Tensor,
        soft_kmeans_iters: int = 15,
        soft_kmeans_temperature: float = 1.0,
        seed: int = 0,
    ) -> None:
        """Initialize learnable tensor pivots from training samples."""
        if tuple(x.shape[1:]) != self.tensor_shape:
            raise ValueError(
                f"Expected x shape [B, {', '.join(map(str, self.tensor_shape))}], "
                f"got {tuple(x.shape)}."
            )
        pivots = soft_kmeans(
            x.detach(),
            num_centers=self.num_pivots,
            num_iters=soft_kmeans_iters,
            temperature=soft_kmeans_temperature,
            seed=seed,
        )
        self.pivots.copy_(pivots)
        self._initialized.fill_(True)

    def _check_input(self, x: Tensor) -> None:
        if x.ndim != 1 + len(self.tensor_shape):
            raise ValueError(
                f"Expected x with {1 + len(self.tensor_shape)} dims including batch, "
                f"got {x.ndim}."
            )
        if tuple(x.shape[1:]) != self.tensor_shape:
            raise ValueError(
                f"Expected tensor shape {self.tensor_shape}, got {tuple(x.shape[1:])}."
            )
        max_rank = min(self.tensor_shape)
        if self.subspace_rank > max_rank:
            raise ValueError(
                f"subspace_rank={self.subspace_rank} exceeds min tensor dimension {max_rank}."
            )
        if not self.initialized:
            raise RuntimeError("Pivots are not initialized. Call initialize_pivots(x) first.")

    def forward(
        self,
        x: Tensor,
        return_stats: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, object]]:
        self._check_input(x)
        x_sub = extract_mode_subspaces(x, self.subspace_rank)
        p_sub = extract_mode_subspaces(self.pivots, self.subspace_rank)

        x_sigmas: List[Tensor]
        p_sigmas: List[Tensor]
        if self.msn is not None:
            x_sigmas = self.msn(x_sub)
            p_sigmas = self.msn(p_sub)
            x_sub = apply_confidence_gate(x_sub, x_sigmas)
            p_sub = apply_confidence_gate(p_sub, p_sigmas)
            sigma_reg = uncertainty_regularizer(x_sigmas)
        else:
            x_sigmas = []
            p_sigmas = []
            sigma_reg = x.new_zeros(())

        k_xc = sum_product_tensor_kernel(
            x_sub,
            p_sub,
            bandwidth=self.bandwidth,
            mu=self.mu,
        )
        k_cc = sum_product_tensor_kernel(
            p_sub,
            p_sub,
            bandwidth=self.bandwidth,
            mu=self.mu,
        )
        features = self.nystrom(k_xc, k_cc)

        if not return_stats:
            return features

        stats: Dict[str, object] = {
            "mu": self.mu.detach(),
            "sigma_regularizer": sigma_reg,
            "input_sigmas": [s.detach() for s in x_sigmas],
            "pivot_sigmas": [s.detach() for s in p_sigmas],
            "kernel_xc": k_xc.detach(),
            "kernel_cc": k_cc.detach(),
        }
        return features, stats


class UKTLClassifier(nn.Module):
    """Small task head on top of the task-agnostic UKTL feature extractor."""

    def __init__(self, feature_extractor: UKTLFeatureExtractor, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        self.features = feature_extractor
        self.classifier = nn.Linear(feature_extractor.num_pivots, num_classes)

    def forward(self, x: Tensor, return_stats: bool = False):
        if return_stats:
            features, stats = self.features(x, return_stats=True)
            logits = self.classifier(features)
            stats["features"] = features.detach()
            return logits, stats
        features = self.features(x)
        return self.classifier(features)
