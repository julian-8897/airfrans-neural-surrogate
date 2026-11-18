"""Compare the pinned authors' AirfRANS Transolver with the local model."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch

from airfoil_surrogate.data.features import TARGET_NAMES, Normaliser
from airfoil_surrogate.data.stage_original import PIN
from airfoil_surrogate.data.staged_original import read_complete_partition
from airfoil_surrogate.models.transolver import Transolver

UPSTREAM_REVISION = "75e0f67643806a81cd1d3f6adc88dd8c02416fe7"
UPSTREAM_SHA256 = "1d0c1b932411ac408d8d00ca047d7f875c3176e7b001db802d30c065a7416d74"
RTOL = 1e-4
ATOL = 1e-5
SEED = 0
SUBSET_SIZE = 4096


def copy_upstream_weights(
    model: Transolver, upstream: dict[str, torch.Tensor]
) -> tuple[dict[str, str], list[str]]:
    """Copy every forward parameter and list the unused upstream MLP weights."""
    mapping = {"placeholder": "placeholder"}
    unused = set()

    def add_pair(source: str, target: str) -> None:
        for field in ("weight", "bias"):
            mapping[f"{source}.{field}"] = f"{target}.{field}"

    add_pair("preprocess.linear_pre.0", "preprocess.0")
    add_pair("preprocess.linear_post", "preprocess.2")
    for i, block in enumerate(model.blocks):
        prefix = f"blocks.{i}"
        for variant in ("ln_1", "ln_2"):
            add_pair(f"{prefix}.{variant}", f"{prefix}.{variant}")
        for key in block.attn.state_dict():
            mapping[f"{prefix}.Attn.{key}"] = f"{prefix}.attn.{key}"
        add_pair(f"{prefix}.mlp.linear_pre.0", f"{prefix}.mlp.0")
        add_pair(f"{prefix}.mlp.linear_post", f"{prefix}.mlp.2")
        # The authors instantiate mlp_new but never call it in forward().
        unused.update(
            f"{prefix}.mlp_new.{layer}.{field}"
            for layer in ("linear_pre.0", "linear_post")
            for field in ("weight", "bias")
        )
    last = f"blocks.{len(model.blocks) - 1}"
    add_pair(f"{last}.ln_3", f"{last}.head.0")
    add_pair(f"{last}.mlp2", f"{last}.head.1")

    expected = set(mapping) | unused
    if expected != set(upstream):
        missing, unexpected = expected - upstream.keys(), upstream.keys() - expected
        raise ValueError(
            f"unexpected upstream state layout: missing {sorted(missing)}, "
            f"unexpected {sorted(unexpected)}"
        )
    if set(mapping.values()) != dict(model.named_parameters()).keys():
        raise ValueError("weight mapping does not cover every local parameter")

    state = model.state_dict()
    for source, target in mapping.items():
        if state[target].dtype != upstream[source].dtype:
            raise ValueError(f"upstream dtype mismatch for {source}")
        state[target] = upstream[source]
    # Retain the local reference grid, which is generated during construction.
    model.load_state_dict(state, strict=True)
    return mapping, sorted(unused)


def compare_outputs(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    """Report all four channel errors and the outcome of the forward check."""
    if (
        reference.ndim != 2
        or reference.shape[1] != len(TARGET_NAMES)
        or actual.shape != reference.shape
    ):
        return {
            "passed": False,
            "error": f"unexpected output shapes {actual.shape}, {reference.shape}",
        }
    if not (torch.isfinite(actual).all() and torch.isfinite(reference).all()):
        return {"passed": False, "error": "non-finite output"}

    delta = actual.double() - reference.double()
    report = {
        "passed": True,
        "max_abs_by_channel": delta.abs().amax(dim=0).tolist(),
        "rms_by_channel": delta.square().mean(dim=0).sqrt().tolist(),
    }
    try:
        torch.testing.assert_close(actual, reference, rtol=RTOL, atol=ATOL)
    except AssertionError as error:
        report.update(passed=False, error=str(error))
    return report


def _load_pinned_reference(path: Path) -> ModuleType:
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("upstream Transolver source checksum mismatch")
    module = ModuleType("upstream_airfrans_transolver")
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


@torch.inference_mode()
def run_parity(
    source: Path, train_stage: Path, normaliser_path: Path, case_id: str
) -> dict:
    from airfoil_surrogate.data.original import load_staged_features

    rows, stage_sha256 = read_complete_partition(
        train_stage, "train", json.loads(PIN.read_text())
    )
    normaliser_bytes = normaliser_path.read_bytes()
    fit = json.loads(normaliser_bytes)
    if fit["fit_stage_sha256"] != stage_sha256 or fit["fit_cases"] != [
        row["id"] for row in rows
    ]:
        raise ValueError("normaliser provenance does not match the training stage")
    cases = {row["id"]: row for row in rows}
    case = cases[case_id]
    features = load_staged_features(train_stage, case)
    if len(features.pos) < 2 * SUBSET_SIZE:
        raise ValueError("training mesh is too small for disjoint parity subsets")

    device = torch.device("cuda", torch.cuda.current_device())
    gpu = torch.cuda.get_device_name(device)
    if "A100" not in gpu or "MIG" in gpu:
        raise RuntimeError(f"the parity run requires a full A100, found {gpu}")
    torch.set_float32_matmul_precision("highest")
    model = Transolver()
    cfg = model.cfg
    upstream = _load_pinned_reference(source)
    torch.manual_seed(SEED)
    reference = upstream.Transolver(
        space_dim=cfg.in_dim,
        n_layers=cfg.n_layers,
        n_hidden=cfg.n_hidden,
        dropout=cfg.dropout,
        n_head=cfg.n_head,
        mlp_ratio=cfg.mlp_ratio,
        fun_dim=0,
        out_dim=cfg.out_dim,
        slice_num=cfg.slice_num,
        ref=cfg.ref,
        unified_pos=True,
    ).eval()
    mapping, unused = copy_upstream_weights(model, reference.state_dict())
    model = model.to(device).eval()
    reference = reference.to(device)

    norm = Normaliser.from_dict(fit)
    x = torch.as_tensor(norm.inputs(features.x), device=device, dtype=torch.float32)
    pos = torch.as_tensor(features.pos, device=device, dtype=torch.float32)
    indices = np.random.default_rng(SEED).choice(
        len(pos), size=2 * SUBSET_SIZE, replace=False
    )
    checks = {}
    for label, selector in (
        ("subset_1", indices[:SUBSET_SIZE]),
        ("subset_2", indices[SUBSET_SIZE:]),
        ("full_mesh", slice(None)),
    ):
        input_x, input_pos = x[selector], pos[selector]
        expected = reference(SimpleNamespace(x=input_x, pos=input_pos))
        actual = model(input_x[None], input_pos[None])[0]
        checks[label] = {"nodes": len(input_pos), **compare_outputs(actual, expected)}

    return {
        "passed": all(check["passed"] for check in checks.values()),
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_sha256": UPSTREAM_SHA256,
        "versions": {key: version(key) for key in ("torch", "timm", "einops")},
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "dtype": "float32",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "seed": SEED,
        "rtol": RTOL,
        "atol": ATOL,
        "case": case,
        "normaliser_sha256": hashlib.sha256(normaliser_bytes).hexdigest(),
        "train_stage_sha256": stage_sha256,
        "channels": TARGET_NAMES,
        "mapped_parameters": mapping,
        "unused_upstream_parameters": unused,
        "local_reference_grid": model.grid.tolist(),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="pinned upstream Transolver.py")
    parser.add_argument("train_stage", type=Path)
    parser.add_argument("normaliser", type=Path)
    parser.add_argument("--case", required=True, help="case ID from the training stage")
    parser.add_argument("--out", type=Path, required=True, help="new JSON report path")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report = run_parity(args.reference, args.train_stage, args.normaliser, args.case)
    with args.out.open("x") as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
