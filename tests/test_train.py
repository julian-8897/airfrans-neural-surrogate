import json
import math
import os
from pathlib import Path

import pytest

from airfoil_surrogate.eval.__main__ import evaluate_run
from airfoil_surrogate.models.transolver import TransolverConfig
from airfoil_surrogate.train import TrainConfig, split_train_val, train

DATA_DIR = Path(
    os.environ.get(
        "AIRFRANS_REMESHED_DIR",
        Path(__file__).resolve().parents[1] / "data/raw/airfrans_remeshed",
    )
)
requires_data = pytest.mark.skipif(not DATA_DIR.exists(), reason="no AirfRANS_remeshed")


@requires_data
def test_validation_is_tail_of_training_list_and_disjoint_from_test():
    from airfoil_surrogate.data.remeshed import load_split_indices

    cfg = TrainConfig(data_dir=str(DATA_DIR))
    train_ids, val_ids = split_train_val(cfg)
    listed = [int(i) for i in load_split_indices(DATA_DIR / "README.md")["full_train"]]
    assert (len(train_ids), len(val_ids)) == (720, 80)
    assert train_ids + val_ids == listed
    test = set(load_split_indices(DATA_DIR / "README.md")["full_test"].tolist())
    assert not (set(train_ids) | set(val_ids)) & test


@requires_data
@pytest.mark.slow
def test_training_loop_writes_run_artefacts(tmp_path):
    cfg = TrainConfig(
        epochs=2,
        val_every=1,
        max_train_cases=3,
        device="cpu",
        data_dir=str(DATA_DIR),
        out_root=str(tmp_path),
        name="test",
        model=TransolverConfig(n_hidden=16, n_layers=1, n_head=2, slice_num=4, ref=2),
    )
    out = train(cfg)
    rows = [
        json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2 and all("val_vol" in r for r in rows)
    assert all(math.isfinite(v) for r in rows for v in r["train_vol"] + r["val_surf"])
    meta = json.loads((out / "run.json").read_text())
    assert len(meta["train_ids"]) == 3 and meta["git"]["rev"]
    assert (out / "model.pt").exists() and (out / "normaliser.json").exists()

    evaluation = evaluate_run(out, tmp_path / "eval", limit=2)
    cases = [
        json.loads(row) for row in (evaluation / "cases.jsonl").read_text().splitlines()
    ]
    assert [case["index"] for case in cases] == meta["val_ids"][:2]
    summary = json.loads((evaluation / "summary.json").read_text())
    assert summary["split"] == "validation"
    assert summary["metrics"]["volume_mse"]["n_defined"] == [2] * 4
