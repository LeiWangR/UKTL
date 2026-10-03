# UKTL

This repository provides a task-agnostic implementation of the UKTL module described in the paper. It operates on arbitrary structured tensors and can be integrated with a user-defined encoder and downstream task head. The action-recognition-specific preprocessing, MLP, HoT backbone, and dataset pipelines are not included.

## Installation

Recommended: Python 3.10+ and a recent PyTorch release.

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

For a GPU installation of PyTorch, install the appropriate CUDA build from pytorch.org before installing the remaining requirements.

## 1-minute demo

The repository includes a small synthetic tensor classification task. It is simple: each class has different low-rank mode subspaces, so the core UKTL mechanism can be exercised without downloading a dataset.

```bash
python train.py --epochs 40
```

Typical output looks like:

```text
epoch 001 | loss 0.9646 | train 58.1% | val 80.0% | mu 0.500
epoch 005 | loss 0.8906 | train 74.4% | val 80.0% | mu 0.500
epoch 010 | loss 0.8339 | train 74.4% | val 80.0% | mu 0.500
epoch 015 | loss 0.7917 | train 74.4% | val 80.0% | mu 0.500
epoch 020 | loss 0.7548 | train 89.5% | val 90.0% | mu 0.500
epoch 025 | loss 0.7218 | train 100.0% | val 100.0% | mu 0.500
epoch 030 | loss 0.6916 | train 100.0% | val 100.0% | mu 0.500
epoch 035 | loss 0.6634 | train 100.0% | val 100.0% | mu 0.500
epoch 040 | loss 0.6366 | train 100.0% | val 100.0% | mu 0.500
saved: runs/demo/best.pt
best validation accuracy: 100.00%
final held-out test accuracy: 91.67%
```

```bash
python test.py --checkpoint runs/demo/best.pt
```

Typical output looks like:

```text
test accuracy: 91.67%
mixture mu: 0.5000
mean sigma per mode: 1.1738, 1.1707, 1.2570
saved: runs/demo/predictions.pt
```

The exact numerical result depends on the installed PyTorch version and hardware, so the demo is intended as a functional smoke test. Model selection uses the validation split; the final test accuracy is reported only after the best validation checkpoint is restored.

## Minimal API

Run the API example from the repository root with:

```bash
python -m examples.api_demo
```

The core object is `UKTLFeatureExtractor`:

```python
import torch
from uktl import UKTLFeatureExtractor

x_train = torch.randn(100, 12, 16, 20)
x_test = torch.randn(20, 12, 16, 20)

uktl = UKTLFeatureExtractor(
    tensor_shape=(12, 16, 20),
    subspace_rank=4,
    num_pivots=32,
    bandwidth=1.0,
    mu_init=0.5,
    learnable_mu=False,  # paper uses mu as a tuned hyperparameter
    uncertainty=True,
)

# Once, using training tensors.
uktl.initialize_pivots(x_train)

features = uktl(x_train)       # [100, 32]
test_features = uktl(x_test)  # [20, 32]
```

You can then attach any task-specific head:

```python
classifier = torch.nn.Linear(features.shape[-1], num_classes)
logits = classifier(features)
```

For segmentation, detection, retrieval, metric learning, regression, or other structured tasks, use the UKTL feature map wherever a tensor-level embedding is appropriate and keep the downstream objective/task architecture separate.


## Controlled KTL vs UKTL experiment

The `examples/` folder includes a controlled synthetic benchmark comparing
**KTL** (the same model with uncertainty disabled) and **UKTL**. The benchmark
is designed to test the setting motivating UKTL: different tensor modes can
have different reliability.

```bash
python -m examples.toy_experiments
```

The experiment uses 4- and 10-class synthetic tensor classification under two
conditions:

1. **Clean subspace task:** all three modes contain class-specific low-rank
   structure and no nuisance component is added. This provides a control
   setting in which all modes are informative.
2. **Nuisance-mode task:** exactly 50% of samples receive a strong,
   sample-dependent higher-rank component varying across mode 3, while modes 1
   and 2 retain the class-related factors. This creates the unequal-mode-
   reliability setting that is the main motivation for the uncertainty
   mechanism.

### Experimental protocol

Each run contains 400 samples with a class-balanced 50% / 20% / 30%
train/validation/test split (200 / 80 / 120 samples).

For the nuisance task, clean and corrupted samples are split separately within
each class, so each of train, validation, and test contains exactly 50% nuisance
samples. KTL and UKTL use the same tensor shape, subspace rank, Nyström pivots,
kernel bandwidth, mixture weight, optimizer, learning rate, weight decay,
gradient clipping, random seeds, and 40 training epochs. Nyström pivots are
initialized from the training split only; validation accuracy is used only for
checkpoint selection; the held-out test set is evaluated after model selection.

### Current results

Results are mean +/- population standard deviation over seeds 0, 1, and 2.

| Task | Classes | Method | Val | Overall test | Clean-subset test | Nuisance-subset test |
|---|---:|---|---:|---:|---:|---:|
| Clean subspace | 4 | KTL | 97.08 +/- 2.57% | 97.22 +/- 1.42% | 97.22 +/- 1.42% | -- |
| Clean subspace | 4 | UKTL | 99.58 +/- 0.59% | 99.17 +/- 0.00% | 99.17 +/- 0.00% | -- |
| Nuisance mode | 4 | KTL | 90.42 +/- 13.55% | 87.22 +/- 16.31% | 85.00 +/- 17.69% | 89.44 +/- 14.93% |
| Nuisance mode | 4 | UKTL | 95.42 +/- 6.48% | 95.83 +/- 3.54% | 95.56 +/- 2.83% | 96.11 +/- 4.37% |
| Clean subspace | 10 | KTL | 73.33 +/- 16.53% | 70.56 +/- 16.31% | 70.56 +/- 16.31% | -- |
| Clean subspace | 10 | UKTL | 88.33 +/- 8.25% | 85.56 +/- 8.17% | 85.56 +/- 8.17% | -- |
| Nuisance mode | 10 | KTL | 55.83 +/- 11.47% | 54.17 +/- 11.24% | 57.22 +/- 13.63% | 51.11 +/- 10.03% |
| Nuisance mode | 10 | UKTL | 71.67 +/- 16.40% | 71.67 +/- 16.72% | 71.67 +/- 18.00% | 71.67 +/- 15.46% |

For the clean task, `Overall test` and `Clean-subset test` are identical because
the test set contains no nuisance samples. For the nuisance task, `Overall
test` is computed over the complete held-out test set, while the two subset
columns isolate the clean and nuisance samples within that same held-out test
set.

The corresponding overall test differences (UKTL minus KTL) are +1.94 points
for 4 classes in the clean task, +8.61 points for 4 classes in the nuisance
task, +15.00 points for 10 classes in the clean task, and +17.50 points for
10 classes in the nuisance task. On the nuisance subset, the UKTL gains are
+6.67 points for 4 classes and +20.56 points for 10 classes.

### Uncertainty diagnostic

UKTL's mean sigma values are:

- 4-class clean: mode 1 = 1.292, mode 2 = 1.292, mode 3 = 1.297.
- 4-class nuisance: mode 1 = 1.176, mode 2 = 1.176, mode 3 = 1.229.
- 10-class clean: mode 1 = 1.333, mode 2 = 1.404, mode 3 = 1.341.
- 10-class nuisance: mode 1 = 1.238, mode 2 = 1.265, mode 3 = 1.343.

In both nuisance-mode experiments, mode 3 has the largest mean sigma, consistent
with the deliberately reduced reliability of that mode. The clean experiments
are a control and do not require the uncertainty scores to favor any particular
mode.

### Interpretation and scope

The nuisance-mode condition is the more direct effectiveness test because it
deliberately creates unequal mode reliability and measures performance on the
corresponding nuisance subset.

The clean-task gains should not be attributed solely to uncertainty: UKTL
contains additional learned MSN parameters. A capacity-matched ablation would
be needed to isolate the contribution of uncertainty weighting from the extra
model capacity. The results therefore demonstrate effectiveness in this
controlled stress setting, not universal superiority over KTL.

## Using your own dataset

For the included `train.py` / `test.py` CLI, save a PyTorch `.pt` file containing a dictionary:

```python
{
    "train_x": train_x,  # [N_train, I1, ..., IM]
    "train_y": train_y,  # [N_train]
    "test_x": test_x,    # [N_test, I1, ..., IM]
    "test_y": test_y,    # [N_test]
}
```

Then run:

```bash
python train.py \
    --data path/to/dataset.pt \
    --output runs/my_task \
    --rank 8 \
    --pivots 64 \
    --bandwidth 1.0 \
    --epochs 50

python test.py \
    --data path/to/dataset.pt \
    --checkpoint runs/my_task/best.pt
```

All samples should share the same tensor dimensions. A batch dimension is added automatically by the model API, so the model expects `[B, I1, ..., IM]` rather than a flattened vector.

## Core modules

```text
uktl/
├── subspace.py      # mode unfolding + top-p SVD bases
├── kernel.py        # Grassmann projection/RBF + sum-product KTL
├── uncertainty.py   # Multi-mode SigmaNet + confidence gating + regularizer
├── nystrom.py       # soft k-means + stable inverse square root + Nyström
├── model.py         # task-agnostic UKTL feature extractor + optional classifier
└── __init__.py
```

### `extract_mode_subspaces`

```python
from uktl import extract_mode_subspaces

subspaces = extract_mode_subspaces(x, rank=8)
# subspaces[m].shape == [B, I_m, 8]
```

### Grassmann kernel

```python
from uktl import sum_product_tensor_kernel

k = sum_product_tensor_kernel(
    subspaces_a,
    subspaces_b,
    bandwidth=1.0,
    mu=0.5,
)
# k.shape == [len(a), len(b)]
```

### MSN architecture 

The repository provides a lightweight reference MSN implementation for uncertainty estimation. The MSN is intentionally modular: its architecture can be adapted to the tensor dimensions, representation, and computational requirements of a particular downstream task, while preserving the UKTL interface of estimating direction-wise uncertainty and using it to reweight the mode-wise subspaces.

### Uncertainty network

```python
from uktl import MultiModeSigmaNet, apply_confidence_gate

msn = MultiModeSigmaNet(
    mode_dims=(12, 16, 20),
    subspace_rank=4,
)
sigmas = msn(subspaces)
weighted = apply_confidence_gate(subspaces, sigmas)
```

## Design notes for research use

**Subspace rank `p`.** This controls how many SVD directions from each mode are retained. Larger `p` captures more mode variation but increases the kernel cost roughly quadratically.

**Number of pivots `C`.** This controls the explicit Nyström feature dimension. Larger `C` improves the approximation but increases kernel and eigendecomposition cost.

**Kernel bandwidth `sigma`.** This is the RBF bandwidth shared across modes, as in the paper.

**Mixture weight `mu`.** The main paper setup describes `mu` as a tunable hyperparameter, while Appendix F.4 describes it as a learnable scalar initialized at 0.5. The reference API keeps it fixed by default to follow the main Eq. (14)/hyperparameter setup; set `learnable_mu=True` or pass `--learnable-mu` to reproduce the Appendix F.4 variant.

**Uncertainty.** Set `uncertainty=False` to recover the non-uncertainty KTL version. The training script uses the batch form of Eq. (22), scaled by `1/N_batch`, together with mean cross-entropy. This preserves the relative weighting of the two terms while keeping the objective scale independent of batch size.

**Centering.** Eq. (20) centers each sample's Nyström feature vector across the pivot dimension. This is implemented directly as `G - G.mean(dim=-1, keepdim=True)`, so test-time centering is deterministic and does not require test-set statistics.

## Numerical stability and exact-vs-robust modes

The paper computes the Nyström inverse square root through the eigendecomposition in Eq. (19). The API therefore supports `method="eigh"` for a literal implementation of that equation.

The default is `method="newton_schulz"`, which computes the same matrix function iteratively without differentiating through eigenvectors. This is a numerical-stability option for learned pivots, not a change to the kernel definition. Use `eigh` when you specifically want the equation-level implementation; use the default for ordinary end-to-end training.

The mode subspaces still come from an SVD. Gradients through singular vectors are inherently sensitive when singular values are repeated or nearly repeated, so no implementation can guarantee finite gradients for every possible tensor. In structured or degenerate inputs, consider regularizing the upstream representation, increasing numerical precision, or treating the subspace extractor as a non-gradient feature map.

The kernel code clips negative squared distances caused only by floating-point round-off.

## Computational considerations

The main costs are:

- mode-wise SVDs; the current reference backend computes a reduced SVD and then retains the leading `p` vectors, so its backend cost is higher than a specialized true truncated/randomized SVD;
- `B x C` kernel evaluations, scaling roughly with `B * C * p^2` after the low-rank projection-distance reduction;
- a `C x C` inverse-square-root transform for the Nyström step.

The paper states that the `C x C` eigendecomposition is updated only every few epochs to amortize its `O(C^3)` cost. This small reference package recomputes the transform in each forward pass for conceptual simplicity and exact differentiability of the chosen backend. For large-scale training, cache/update the pivot transform less frequently, as done in the paper.

For large tensors, avoid explicitly materializing `U U^T` outside the uncertainty network. The kernel implementation already uses the low-rank form.

The paper reports that practical performance saturated around `p=8-10` and approximately `C=150-180` on its action-recognition experiments. Those values are dataset-specific and should not be treated as universal defaults.

## Tests

```bash
pytest
```

The tests cover unfolding shapes, SVD orthonormality, explicit-vs-low-rank projection-distance equivalence, projection-kernel symmetry/PSD behavior, Eq. (22) loss algebra, Nyström inverse-square-root consistency, full forward/backward gradients in a non-degenerate case, and the uncertainty-disabled variant.

## Notes

**Spectral gradients.** PyTorch documents that gradients through SVD singular vectors can become unstable or undefined around repeated/nearby singular values. The same issue applies to eigenvectors in `torch.linalg.eigh`. The robust Nyström default therefore avoids eigenvector backpropagation, but the SVD subspace extraction necessarily remains a spectral operation.

**Uncertainty loss.** Eq. (22) is implemented literally by default. Its denominator is the sum over the current samples for each mode/direction, exactly as written. The Gaussian derivation in Appendix instead gives a weighted residual term plus a `log sigma` term. These two objectives are not algebraically identical, so the implementation should not describe Eq. (22) as a direct MLE consequence without qualification. In fact, for `n>1` samples with all `sigma_i=c`, its regularizer is `n log((c+1)/(n c+1))`, which decreases as `c` grows. Thus its standalone minimization pushes the bounded sigmas upward, contrary to calling it a conventional anti-explosion barrier. This does not invalidate the kernel construction.

**Uncertainty changes the geometry.** Before uncertainty weighting, `U U^T` is an orthogonal projector representing a Grassmann point. After column-wise scaling by `1/sqrt(sigma)`, the columns are no longer generally orthonormal, so the weighted representation is not literally another point on the same Grassmann manifold. The resulting kernel is still well-defined through its deterministic finite-dimensional embedding, but it is more precise to call it a confidence-weighted projection embedding than an unchanged Grassmann distance.

## Scope

This repository implements the reusable **algorithmic core**, not the complete action-recognition experiment stack, preprocessing, benchmark loaders, or backbone used in the paper. That separation is intentional: the same UKTL tensor-kernel module can be integrated with a domain-specific encoder for other structured data. The included synthetic example is a correctness/demo harness. 

## Citation

If this implementation is useful in your work, please cite the UKTL paper:

```bibtex
@inproceedings{
wang2026subspace,
title={Subspace Kernel Learning on Tensor Sequences},
author={Lei Wang and Xi Ding and Yongsheng Gao and Piotr Koniusz},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=kv22NbU2T2}
}
```
