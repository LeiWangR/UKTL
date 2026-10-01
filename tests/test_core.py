import torch

from uktl import (
    UKTLFeatureExtractor,
    extract_mode_subspaces,
    inverse_sqrt_psd,
    mode_unfold,
    squared_projection_distance,
    sum_product_tensor_kernel,
    uncertainty_regularizer,
)


def test_mode_unfold_shape():
    x = torch.randn(2, 3, 4, 5)
    assert mode_unfold(x, 0).shape == (2, 3, 20)
    assert mode_unfold(x, 1).shape == (2, 4, 15)
    assert mode_unfold(x, 2).shape == (2, 5, 12)


def test_subspace_shapes_and_orthonormality():
    x = torch.randn(3, 5, 6, 7)
    subs = extract_mode_subspaces(x, rank=3)
    assert [u.shape for u in subs] == [(3, 5, 3), (3, 6, 3), (3, 7, 3)]
    for u in subs:
        gram = u.transpose(-1, -2) @ u
        eye = torch.eye(3).expand_as(gram)
        assert torch.allclose(gram, eye, atol=1e-4, rtol=1e-4)


def test_projection_distance_matches_explicit_projector_formula():
    torch.manual_seed(0)
    a = torch.linalg.qr(torch.randn(3, 7, 2, dtype=torch.float64), mode="reduced").Q
    b = torch.linalg.qr(torch.randn(4, 7, 2, dtype=torch.float64), mode="reduced").Q
    d_fast = squared_projection_distance(a, b)
    pa = a @ a.transpose(-1, -2)
    pb = b @ b.transpose(-1, -2)
    d_ref = (pa[:, None] - pb[None, :]).square().sum(dim=(-1, -2))
    assert torch.allclose(d_fast, d_ref, atol=1e-10, rtol=1e-10)


def test_sum_product_kernel_is_symmetric_psd_on_small_set():
    x = torch.randn(8, 5, 6, 7)
    subs = extract_mode_subspaces(x, rank=2)
    k = sum_product_tensor_kernel(subs, subs, bandwidth=1.0, mu=0.5)
    assert k.shape == (8, 8)
    assert torch.allclose(k, k.T, atol=1e-5)
    eigvals = torch.linalg.eigvalsh((k + k.T) / 2)
    assert float(eigvals.min()) > -1e-4


def test_uncertainty_regularizer_exact_sum_and_mean():
    s1 = torch.tensor([[0.5, 1.0], [1.5, 2.0]])
    s2 = torch.tensor([[2.0, 0.5], [1.0, 1.5]])
    exact = 0.0
    for s in (s1, s2):
        denom = s.sum(dim=0, keepdim=True) + 1.0
        exact += torch.log((s + 1.0) / denom).sum().item()
    got = uncertainty_regularizer((s1, s2), eps=0.0, reduction="sum").item()
    mean = uncertainty_regularizer((s1, s2), eps=0.0, reduction="mean").item()
    assert abs(got - exact) < 1e-7
    assert abs(mean - exact / 8.0) < 1e-7


def test_inverse_sqrt_newton_matches_eigh():
    torch.manual_seed(0)
    a = torch.randn(6, 6, dtype=torch.float64)
    a = a @ a.T + 0.2 * torch.eye(6, dtype=torch.float64)
    ref = inverse_sqrt_psd(a, jitter=1e-5, method="eigh", eps=1e-10)
    ns = inverse_sqrt_psd(a, jitter=1e-5, method="newton_schulz", eps=1e-10, iterations=40)
    assert torch.allclose(ns, ref, atol=2e-7, rtol=2e-6)


def test_full_model_forward_backward():
    torch.manual_seed(0)
    x = torch.randn(12, 5, 6, 7, requires_grad=True)
    y = torch.randint(0, 3, (12,))
    features = UKTLFeatureExtractor(
        tensor_shape=(5, 6, 7),
        subspace_rank=2,
        num_pivots=5,
        bandwidth=1.0,
        uncertainty=True,
        nystrom_inverse_sqrt_method="newton_schulz",
    )
    features.initialize_pivots(x, soft_kmeans_iters=3)
    classifier = torch.nn.Linear(5, 3)
    logits, stats = features(x, return_stats=True)
    logits = classifier(logits)
    loss = torch.nn.functional.cross_entropy(logits, y) + 0.01 * stats["sigma_regularizer"] / y.numel()
    loss.backward()
    assert torch.isfinite(loss)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert features.pivots.grad is not None and torch.isfinite(features.pivots.grad).all()
    for parameter in list(features.parameters()) + list(classifier.parameters()):
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_no_uncertainty_variant():
    x = torch.randn(6, 4, 5, 6)
    features = UKTLFeatureExtractor(
        tensor_shape=(4, 5, 6),
        subspace_rank=2,
        num_pivots=4,
        uncertainty=False,
    )
    features.initialize_pivots(x, soft_kmeans_iters=2)
    out = features(x)
    assert out.shape == (6, 4)


def test_learnable_mu_receives_gradient():
    torch.manual_seed(1)
    x = torch.randn(8, 4, 5, 6)
    y = torch.randint(0, 2, (8,))
    features = UKTLFeatureExtractor(
        tensor_shape=(4, 5, 6),
        subspace_rank=2,
        num_pivots=4,
        learnable_mu=True,
        uncertainty=False,
    )
    features.initialize_pivots(x, soft_kmeans_iters=2)
    head = torch.nn.Linear(4, 2)
    logits = head(features(x))
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    assert features.raw_mu.grad is not None
    assert torch.isfinite(features.raw_mu.grad).all()


def test_eigh_nystrom_backward_in_generic_case():
    torch.manual_seed(2)
    x = torch.randn(10, 4, 5, 6, requires_grad=True)
    y = torch.randint(0, 3, (10,))
    features = UKTLFeatureExtractor(
        tensor_shape=(4, 5, 6),
        subspace_rank=2,
        num_pivots=5,
        uncertainty=False,
        nystrom_inverse_sqrt_method="eigh",
    )
    features.initialize_pivots(x.detach(), soft_kmeans_iters=2)
    head = torch.nn.Linear(5, 3)
    loss = torch.nn.functional.cross_entropy(head(features(x)), y)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(features.pivots.grad).all()
