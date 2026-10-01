# UKTL

This repository provides a task-agnostic implementation of the UKTL module described in the paper. It operates on arbitrary structured tensors and can be integrated with a user-defined encoder and downstream task head. The action-recognition-specific preprocessing, MLP, HoT backbone, and dataset pipelines are not included.

## Installation

Recommended: Python 3.10+ and a recent PyTorch release.

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\\Scripts\\activate       # Windows PowerShell

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


## Toy effectiveness experiment

`examples/toy_experiments.py` provides a controlled comparison between **KTL** (the same model with uncertainty disabled) and **UKTL**. The synthetic tensors are generated so that classes share the same tensor size while their discriminative information is encoded in mode-wise low-rank subspaces. A second setting makes one tensor mode a class-independent nuisance mode, which is designed to test the purpose of uncertainty weighting.

The script uses a train/validation/test split for every seed: pivots are initialized from training data only, checkpoints are selected using validation accuracy only, and the test set is evaluated once after selection. It reports mean ± standard deviation across multiple seeds and the learned mean sigma per mode. These are illustrative sanity experiments.

Run:

```bash
python -m examples.toy_experiments
```

A representative run with the included seeds currently gives:

```text
Clean subspace task
KTL          test = 96.04 ± 3.40%
             val  = 98.96 ± 1.47%
UKTL         test = 95.00 ± 1.84%
             val  = 98.96 ± 1.47%
             mean sigma = 1.170, 1.193, 1.215

Nuisance-mode task
KTL          test = 96.46 ± 0.59%
             val  = 96.88 ± 2.55%
UKTL         test = 95.00 ± 3.10%
             val  = 98.96 ± 1.47%
             mean sigma = 1.047, 1.059, 1.228
Higher sigma in the nuisance mode is an intended qualitative diagnostic, not a target value.
```

**Controlled toy experiments.** These experiments are designed to probe the behavior of the UKTL components rather than to reproduce the benchmark results in the paper. The clean-subspace task verifies that the subspace kernel can already provide strong discrimination when all modes are informative. The nuisance-mode task introduces a deliberately class-independent, randomly varying third mode and examines whether MSN assigns higher uncertainty to that mode. A higher uncertainty score serves as a qualitative diagnostic of mode-aware weighting and does not imply that UKTL must achieve higher classification accuracy on every synthetic task.

**Interpretation.** In the clean setting, the three mean uncertainty values are relatively close (1.170, 1.193, and 1.215), and KTL slightly outperforms UKTL in this particular three-seed run (96.04% vs. 95.00% test accuracy). This is consistent with the setting in which all modes are informative, where additional uncertainty weighting is not necessarily beneficial. In the nuisance-mode setting, MSN assigns the largest mean uncertainty to the deliberately class-independent mode 3 (1.228), compared with 1.047 and 1.059 for modes 1 and 2. This indicates that MSN can differentiate the relative reliability of tensor modes in this controlled setting. The lower UKTL accuracy in this toy experiment does not contradict this diagnostic: uncertainty estimation and downstream classification performance are related but distinct objectives.

**Summary.** KTL provides the structured subspace kernel, while UKTL augments it with adaptive uncertainty estimation. These toy experiments illustrate the two mechanisms separately; the benchmark experiments in the paper evaluate their effect on real recognition tasks.

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
