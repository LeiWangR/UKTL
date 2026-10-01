"""UKTL: Uncertainty-driven Kernel Tensor Learning."""

from .kernel import grassmann_rbf_kernel, squared_projection_distance, sum_product_tensor_kernel
from .model import UKTLClassifier, UKTLFeatureExtractor
from .nystrom import NystromLinearizer, inverse_sqrt_psd, soft_kmeans
from .subspace import extract_mode_subspaces, mode_unfold, projection_matrices
from .uncertainty import MultiModeSigmaNet, apply_confidence_gate, confidence_gate, uncertainty_regularizer

__all__ = [
    "UKTLFeatureExtractor",
    "UKTLClassifier",
    "extract_mode_subspaces",
    "mode_unfold",
    "projection_matrices",
    "grassmann_rbf_kernel",
    "squared_projection_distance",
    "sum_product_tensor_kernel",
    "MultiModeSigmaNet",
    "confidence_gate",
    "apply_confidence_gate",
    "uncertainty_regularizer",
    "soft_kmeans",
    "inverse_sqrt_psd",
    "NystromLinearizer",
]
