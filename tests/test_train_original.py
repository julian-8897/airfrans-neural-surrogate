import hashlib
import json

import numpy as np
import pytest
import torch

from airfoil_surrogate import train_original as runner
from airfoil_surrogate.data.features import NodeFeatures, Normaliser
from airfoil_surrogate.models.transolver import TransolverConfig


def test_sample_keeps_every_wall_node_and_draws_unique_off_wall_nodes():
    vertices = torch.arange(18, dtype=torch.float32)
    pos = torch.column_stack((vertices, vertices + 1))
    y = torch.column_stack([vertices + i for i in range(4)])
    case = runner.PackedCase(
        x=vertices[:, None].expand(-1, 7),
        pos=pos,
        y=y,
        wall=np.array([2, 6, 15]),
        off_wall=np.delete(np.arange(18), [2, 6, 15]),
    )
    sample = case.sample(0, 10, np.random.default_rng(0), torch.device("cpu"))
    seen = sample.pos[0, :, 0].tolist()
    assert len(set(seen)) == len(seen) == 10
    assert set(sample.pos[0, sample.surf, 0].tolist()) == {2, 6, 15}
    torch.testing.assert_close(sample.x[0, :, 0], sample.pos[0, :, 0], rtol=0, atol=0)
    torch.testing.assert_close(sample.y[0, :, 0], sample.pos[0, :, 0], rtol=0, atol=0)


def test_resume_matches_uninterrupted_training_and_preserves_prior_attempt(
    tmp_path, monkeypatch
):
    rng = np.random.default_rng(5)
    raw_cases = [
        NodeFeatures(
            pos=rng.normal(size=(16, 2)),
            x=rng.normal(size=(16, 7)),
            y=rng.normal(size=(16, 4)),
            surf=np.arange(16) < 3,
        )
        for _ in range(5)
    ]
    norm = Normaliser.fit(raw_cases[:3])
    cases = [runner.PackedCase.pack(features, norm) for features in raw_cases]
    cfg = runner.OriginalTrainConfig(
        epochs=4,
        val_every=2,
        sample_nodes=7,
        model=TransolverConfig(
            n_hidden=8, n_layers=1, n_head=2, slice_num=4, ref=2, dropout=0.15
        ),
    )
    kwargs = {
        "device": torch.device("cpu"),
        "normaliser_bytes": json.dumps(norm.to_dict()).encode(),
        "inputs": {
            "train_stage_sha256": "test-train",
            "validation_stage_sha256": "val",
        },
    }
    uninterrupted = tmp_path / "full"
    runner.train_original(cfg, cases[:3], cases[3:], out=uninterrupted, **kwargs)

    save = runner._save_resume

    def interrupt_after_checkpoint(state, path):
        save(state, path)
        if state["next_epoch"] == 2:
            raise InterruptedError("interrupted after validation")

    interrupted = tmp_path / "interrupted"
    with monkeypatch.context() as patch:
        patch.setattr(runner, "_save_resume", interrupt_after_checkpoint)
        with pytest.raises(InterruptedError):
            runner.train_original(cfg, cases[:3], cases[3:], out=interrupted, **kwargs)

    checkpoint = interrupted / "resume.pt"
    prior_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert not (interrupted / "selection.json").exists()
    resumed = tmp_path / "resumed"
    runner.train_original(
        cfg, cases[:3], cases[3:], out=resumed, resume=checkpoint, **kwargs
    )
    assert json.loads((resumed / "run.json").read_text())["start_epoch"] == 2
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == prior_sha

    def load_weights(path):
        return torch.load(path / "model.pt", weights_only=True)["model"]

    torch.testing.assert_close(
        load_weights(uninterrupted), load_weights(resumed), rtol=0, atol=0
    )
    full_state = torch.load(uninterrupted / "resume.pt", weights_only=True)
    resumed_state = torch.load(resumed / "resume.pt", weights_only=True)
    torch.testing.assert_close(
        full_state["model"], resumed_state["model"], rtol=0, atol=0
    )

    def metrics(path):
        rows = (path / "metrics.jsonl").read_text().splitlines()
        return [
            {k: v for k, v in json.loads(row).items() if k != "seconds"} for row in rows
        ]

    assert [row["epoch"] for row in metrics(resumed)] == [1, 2, 3, 4]
    assert metrics(resumed) == metrics(uninterrupted)
    selected = json.loads((resumed / "selection.json").read_text())
    assert [item["epoch"] for item in selected["validation_scores"]] == [2, 4]
    assert (
        selected["epoch"]
        == min(
            selected["validation_scores"],
            key=lambda item: (item["score"], item["epoch"]),
        )["epoch"]
    )
    assert (
        selected["checkpoint_sha256"]
        == hashlib.sha256((resumed / "model.pt").read_bytes()).hexdigest()
    )

    incompatible = kwargs | {"inputs": {"train_stage_sha256": "changed"}}
    rejected = tmp_path / "rejected"
    with pytest.raises(ValueError, match="different training inputs"):
        runner.train_original(
            cfg, cases[:3], cases[3:], out=rejected, resume=checkpoint, **incompatible
        )
    assert not rejected.exists()


def test_preflight_scales_training_but_not_full_validation_timings():
    from airfoil_surrogate.preflight_original import project_training

    rows = [
        {"seconds": 4.0},
        {"seconds": 2.0},
        {"seconds": 3.0},
        {"seconds": 8.0, "validation_score": 1.0},
    ]
    inputs = {
        "trial_cases": 12,
        "full_cases": 720,
        "training_load_seconds": 1.5,
        "validation_load_seconds": 2.0,
        "trial_seconds": 21.0,
        "config": runner.OriginalTrainConfig(),
    }
    projection = project_training(rows, **inputs)
    assert projection["components_seconds"] == {
        "training_updates": 71_640.0,
        "validation": 220.0,
        "loading": 92.0,
        "outside_epoch": 398.0,
    }
    assert projection["total_seconds"] == 72_350.0
    assert projection["within_training_reservation"]

    rows[2]["seconds"] = 4.0
    assert not project_training(rows, **inputs)["within_training_reservation"]
