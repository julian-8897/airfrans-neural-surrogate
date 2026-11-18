import pytest
import torch
from torch import nn

from airfoil_surrogate.models.transolver import (
    PhysicsAttention,
    Transolver,
    TransolverConfig,
)
from airfoil_surrogate.models.transolver_parity import (
    _load_pinned_reference,
    compare_outputs,
    copy_upstream_weights,
)


def test_copy_upstream_airfrans_layout_and_keep_generated_grid():
    config = TransolverConfig(n_layers=1, n_hidden=16, n_head=4, slice_num=8, ref=3)
    model = Transolver(config)
    grid = model.grid.clone()
    ref = nn.Module()
    ref.register_parameter("placeholder", nn.Parameter(torch.ones(16)))
    ref.preprocess = nn.Module()
    ref.preprocess.linear_pre = nn.Sequential(nn.Linear(16, 32), nn.GELU())
    ref.preprocess.linear_post = nn.Linear(32, 16)

    block = nn.Module()
    block.Attn = PhysicsAttention(16, 4, 8, 0.0)
    block.ln_1 = nn.LayerNorm(16)
    block.ln_2 = nn.LayerNorm(16)
    block.ln_3 = nn.LayerNorm(16)
    block.mlp = nn.Module()
    block.mlp.linear_pre = nn.Sequential(nn.Linear(16, 32), nn.GELU())
    block.mlp.linear_post = nn.Linear(32, 16)
    block.mlp_new = nn.Module()
    block.mlp_new.linear_pre = nn.Sequential(nn.Linear(16, 32), nn.GELU())
    block.mlp_new.linear_post = nn.Linear(32, 16)
    block.mlp2 = nn.Linear(16, 4)
    ref.blocks = nn.ModuleList([block])
    with torch.no_grad():
        for i, parameter in enumerate(ref.parameters()):
            parameter.fill_(i + 1)

    state = ref.state_dict()
    mapping, unused = copy_upstream_weights(model, state)
    assert set(unused) == {
        "blocks.0.mlp_new.linear_pre.0.weight",
        "blocks.0.mlp_new.linear_pre.0.bias",
        "blocks.0.mlp_new.linear_post.weight",
        "blocks.0.mlp_new.linear_post.bias",
    }
    assert len(mapping) == len(list(model.parameters()))
    torch.testing.assert_close(model.placeholder, ref.placeholder, rtol=0, atol=0)
    torch.testing.assert_close(model.grid, grid, rtol=0, atol=0)
    for actual, expected in (
        (model.preprocess[0].weight, ref.preprocess.linear_pre[0].weight),
        (model.preprocess[2].bias, ref.preprocess.linear_post.bias),
        (model.blocks[0].attn.to_v.weight, block.Attn.to_v.weight),
        (model.blocks[0].mlp[2].weight, block.mlp.linear_post.weight),
        (model.blocks[0].head[0].weight, block.ln_3.weight),
        (model.blocks[0].head[1].bias, block.mlp2.bias),
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    state["extra.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unexpected upstream state layout"):
        copy_upstream_weights(model, state)


def test_parity_reports_a_single_channel_mismatch():
    reference = torch.zeros(2, 4)
    assert compare_outputs(reference, reference)["passed"]
    actual = reference.clone()
    actual[0, 2] = 0.125

    report = compare_outputs(actual, reference)
    assert not report["passed"]
    assert report["max_abs_by_channel"] == [0, 0, 0.125, 0]
    assert report["rms_by_channel"] == pytest.approx([0, 0, 0.125 / 2**0.5, 0])


def test_parity_rejects_unpinned_source_before_importing(tmp_path):
    path = tmp_path / "Transolver.py"
    path.write_text("raise RuntimeError('unexpected import')\n")
    with pytest.raises(ValueError, match="source checksum mismatch"):
        _load_pinned_reference(path)
