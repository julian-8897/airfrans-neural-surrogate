import torch

from airfoil_surrogate.models.transolver import Transolver, TransolverConfig

SMALL = TransolverConfig(n_hidden=32, n_layers=2, n_head=4, slice_num=8, ref=4)


def inputs(n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(1, n, 2, generator=g) * torch.tensor([6.0, 3.0]) - torch.tensor(
        [2.0, 1.5]
    )
    x = torch.randn(1, n, 7, generator=g)
    return x, pos


def model(cfg=SMALL, seed=0):
    torch.manual_seed(seed)
    return Transolver(cfg).eval()


def test_output_shape_dtype_and_finite_for_variable_node_counts():
    m = model()
    for n in (1, 37, 500):
        x, pos = inputs(n)
        y = m(x, pos)
        assert y.shape == (1, n, 4) and y.dtype == torch.float32
        assert torch.isfinite(y).all()


def test_permutation_equivariant_over_nodes():
    # Slicing, slice attention and de-slicing commute with node reordering.
    m = model()
    x, pos = inputs(300)
    perm = torch.randperm(300, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        y = m(x, pos)
        y_perm = m(x[:, perm], pos[:, perm])
    torch.testing.assert_close(y_perm, y[:, perm], rtol=1e-5, atol=1e-6)


def test_batch_entries_are_independent():
    m = model()
    x1, p1 = inputs(200, seed=1)
    x2, p2 = inputs(200, seed=2)
    with torch.no_grad():
        both = m(torch.cat([x1, x2]), torch.cat([p1, p2]))
        torch.testing.assert_close(both[1:], m(x2, p2), rtol=1e-5, atol=1e-6)


def test_fixed_seed_gives_identical_outputs():
    x, pos = inputs(100)
    with torch.no_grad():
        torch.testing.assert_close(model(seed=3)(x, pos), model(seed=3)(x, pos))


def test_every_parameter_receives_gradient():
    m = model().train()
    x, pos = inputs(64)
    m(x, pos).square().mean().backward()
    no_grad = [n for n, p in m.named_parameters() if p.grad is None]
    assert not no_grad


def test_reference_grid_spans_domain():
    m = model(TransolverConfig(ref=8))
    assert m.grid.shape == (64, 2)
    assert m.grid[:, 0].min() == -2.0 and m.grid[:, 0].max() == 4.0
    assert m.grid[:, 1].min() == -1.5 and m.grid[:, 1].max() == 1.5
