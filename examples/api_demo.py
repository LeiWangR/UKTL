"""Minimal API example for researchers integrating UKTL into another task."""

import torch

from uktl import UKTLFeatureExtractor


# Suppose a task encoder already produces tensors [batch, I1, I2, I3].
x = torch.randn(8, 12, 16, 20)

uktl = UKTLFeatureExtractor(
    tensor_shape=(12, 16, 20),
    subspace_rank=4,
    num_pivots=6,
    bandwidth=1.0,
    mu_init=0.5,
    learnable_mu=True,
    uncertainty=True,
)

# Initialize pivots once from training tensors.
uktl.initialize_pivots(x)

# Explicit UKTL representation [batch, num_pivots].
features, stats = uktl(x, return_stats=True)
print("features:", features.shape)
print("learned sum/product weight:", float(stats["mu"]))

# Attach any downstream head suitable for the user's task.
head = torch.nn.Linear(features.shape[-1], 5)
logits = head(features)
print("logits:", logits.shape)
