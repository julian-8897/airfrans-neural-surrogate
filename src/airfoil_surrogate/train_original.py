"""Train and resume the full-task Transolver on original AirfRANS meshes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from airfoil_surrogate.data.features import NodeFeatures, Normaliser
from airfoil_surrogate.data.stage_original import PIN
from airfoil_surrogate.data.staged_original import read_complete_partition
from airfoil_surrogate.gpu_budget import TRAINING_MAX_SECONDS
from airfoil_surrogate.models.transolver import Transolver, TransolverConfig
from airfoil_surrogate.models.transolver_parity import verified_parity
from airfoil_surrogate.train import Case, evaluate, field_losses, git_state


@dataclass(frozen=True)
class OriginalTrainConfig:
    epochs: int = 398
    lr: float = 1e-3
    seed: int = 0
    sample_nodes: int = 32_000
    val_every: int = 10
    model: TransolverConfig = field(default_factory=TransolverConfig)


@dataclass(frozen=True)
class PackedCase:
    x: torch.Tensor
    pos: torch.Tensor
    y: torch.Tensor
    wall: np.ndarray
    off_wall: np.ndarray

    @classmethod
    def pack(cls, features: NodeFeatures, norm: Normaliser) -> PackedCase:
        return cls(
            torch.as_tensor(norm.inputs(features.x), dtype=torch.float32),
            torch.as_tensor(features.pos, dtype=torch.float32),
            torch.as_tensor(norm.targets(features.y), dtype=torch.float32),
            np.flatnonzero(features.surf),
            np.flatnonzero(~features.surf),
        )

    def sample(
        self, index: int, count: int, rng: np.random.Generator, device: torch.device
    ) -> Case:
        count_off_wall = count - len(self.wall)
        selected = np.concatenate(
            (self.wall, rng.choice(self.off_wall, count_off_wall, replace=False))
        )
        mask = torch.arange(count, device=device) < len(self.wall)
        return Case(
            index,
            self.x[selected][None].to(device),
            self.pos[selected][None].to(device),
            self.y[selected][None].to(device),
            mask,
        )


def _save_resume(state: dict, path: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
    try:
        torch.save(state, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def train_original(
    cfg: OriginalTrainConfig,
    training: list[PackedCase],
    validation: list[PackedCase],
    *,
    device: torch.device,
    out: Path,
    normaliser_bytes: bytes,
    inputs: dict,
    resume: Path | None = None,
) -> Path:
    """Train an attempt with a new output directory and optional prior state."""
    for case in (*training, *validation):
        if (
            not 0 < len(case.wall) < cfg.sample_nodes
            or len(case.wall) + len(case.off_wall) < cfg.sample_nodes
        ):
            raise ValueError("each mesh must provide a wall and enough off-wall nodes")

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    val_rng = np.random.default_rng(cfg.seed)
    fixed_val = [
        case.sample(i, cfg.sample_nodes, val_rng, device)
        for i, case in enumerate(validation)
    ]
    model = Transolver(cfg.model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * len(training)
    )
    identity = {
        "config": asdict(cfg),
        "inputs": inputs,
        "normaliser_sha256": hashlib.sha256(normaliser_bytes).hexdigest(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda if device.type == "cuda" else None,
        "device": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
    }

    start = 0
    rows = []
    best = None
    if resume is not None:
        previous = torch.load(resume, map_location="cpu", weights_only=True)
        if previous["identity"] != identity:
            raise ValueError(
                "resume checkpoint is for different training inputs or settings"
            )
        start, rows, best = previous["next_epoch"], previous["rows"], previous["best"]
        model.load_state_dict(previous["model"])
        opt.load_state_dict(previous["optimiser"])
        scheduler.load_state_dict(previous["scheduler"])
        rng.bit_generator.state = previous["numpy_rng"]
        torch.set_rng_state(previous["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(previous["cuda_rng"], device)

    out.mkdir(parents=True, exist_ok=False)
    with (out / "run.json").open("x") as stream:
        json.dump(
            identity
            | {
                "git": git_state(),
                "start_epoch": start,
                "resume_sha256": _sha256(resume) if resume is not None else None,
            },
            stream,
            indent=2,
        )
    (out / "normaliser.json").write_bytes(normaliser_bytes)

    for epoch in range(start, cfg.epochs):
        model.train()
        t0 = time.perf_counter()
        vol_sum = torch.zeros(cfg.model.out_dim, device=device)
        surf_sum = torch.zeros_like(vol_sum)
        for i in rng.permutation(len(training)):
            case = training[i].sample(int(i), cfg.sample_nodes, rng, device)
            vol, surf = field_losses(model(case.x, case.pos), case)
            opt.zero_grad(set_to_none=True)
            (vol.mean() + surf.mean()).backward()
            opt.step()
            scheduler.step()
            vol_sum += vol.detach()
            surf_sum += surf.detach()

        train_vol = (vol_sum / len(training)).tolist()
        train_surf = (surf_sum / len(training)).tolist()
        if not np.isfinite(train_vol + train_surf).all():
            raise RuntimeError(f"non-finite training loss at epoch {epoch + 1}")
        row = {
            "epoch": epoch + 1,
            "lr": scheduler.get_last_lr()[0],
            "train_vol": train_vol,
            "train_surf": train_surf,
        }
        if (epoch + 1) % cfg.val_every == 0 or epoch + 1 == cfg.epochs:
            row |= evaluate(model, fixed_val)
            score = float(np.mean(row["val_vol"]) + np.mean(row["val_surf"]))
            if not np.isfinite(score):
                raise RuntimeError(f"non-finite validation loss at epoch {epoch + 1}")
            row["validation_score"] = score
            if best is None or score < best["score"]:
                best = {
                    "epoch": epoch + 1,
                    "score": score,
                    "model": {
                        name: value.detach().to("cpu", copy=True)
                        for name, value in model.state_dict().items()
                    },
                }
        row["seconds"] = time.perf_counter() - t0
        rows.append(row)
        _save_resume(
            {
                "identity": identity,
                "next_epoch": epoch + 1,
                "model": model.state_dict(),
                "optimiser": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "numpy_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": (
                    torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                ),
                "rows": rows,
                "best": best,
            },
            out / "resume.pt",
        )
        print(json.dumps(row, allow_nan=False), flush=True)

    checkpoint = out / "model.pt"
    with checkpoint.open("xb") as stream:
        torch.save({"model": best["model"], "config": asdict(cfg.model)}, stream)
    with (out / "metrics.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    with (out / "selection.json").open("x") as stream:
        json.dump(
            {
                "epoch": best["epoch"],
                "score": best["score"],
                "checkpoint_sha256": _sha256(checkpoint),
                "validation_scores": [
                    {"epoch": row["epoch"], "score": row["validation_score"]}
                    for row in rows
                    if "validation_score" in row
                ],
            },
            stream,
            indent=2,
        )
    return checkpoint


def full_a100() -> torch.device:
    if torch.cuda.device_count() != 1:
        raise RuntimeError("original-mesh training requires one visible full A100")
    device = torch.device("cuda:0")
    gpu = torch.cuda.get_device_name(device)
    if "A100" not in gpu or "MIG" in gpu:
        raise RuntimeError(f"expected a full A100, found {gpu}")
    return device


def training_inputs(
    train_stage: Path, validation_stage: Path, normaliser: Path
) -> tuple[list[dict], list[dict], bytes, dict]:
    pin = json.loads(PIN.read_text())
    train_rows, train_sha256 = read_complete_partition(train_stage, "train", pin)
    val_rows, val_sha256 = read_complete_partition(validation_stage, "validation", pin)
    normaliser_bytes = normaliser.read_bytes()
    fit = json.loads(normaliser_bytes)
    if fit["fit_stage_sha256"] != train_sha256 or fit["fit_cases"] != [
        row["id"] for row in train_rows
    ]:
        raise ValueError("normaliser provenance does not match the training stage")
    return (
        train_rows,
        val_rows,
        normaliser_bytes,
        {
            "train_stage_sha256": train_sha256,
            "validation_stage_sha256": val_sha256,
        },
    )


def pack_cases(stage: Path, rows: list[dict], norm: Normaliser) -> list[PackedCase]:
    from airfoil_surrogate.data.original import load_staged_features

    return [PackedCase.pack(load_staged_features(stage, row), norm) for row in rows]


def check_training_qualification(
    root: Path,
    inputs: dict,
    normaliser_bytes: bytes,
    config: OriginalTrainConfig,
    environment: dict,
) -> dict[str, str]:
    """Bind the launch and any recovery to the successful qualification reports."""
    norm_sha = hashlib.sha256(normaliser_bytes).hexdigest()
    parity, parity_sha = verified_parity(
        root / "transolver-parity.json", inputs["train_stage_sha256"], norm_sha
    )
    raw = (root / "training-preflight" / "preflight.json").read_bytes()
    preflight = json.loads(raw)
    projection = preflight["projection"]
    if (
        preflight["schema"] != 1
        or preflight["inputs"] != inputs
        or preflight["normaliser_sha256"] != norm_sha
        or preflight["parity_sha256"] != parity_sha
        or preflight["production_config"] != json.loads(json.dumps(asdict(config)))
        or preflight["environment"] != environment
        or parity["gpu"] != environment["gpu"]
        or parity["versions"]["torch"] != environment["torch"]
        or parity["cuda"] != environment["cuda"]
        or projection["within_training_reservation"] is not True
        or not 0 < projection["total_seconds"] <= TRAINING_MAX_SECONDS
    ):
        raise ValueError("training requires a matching successful A100 preflight")
    return {
        "parity_sha256": parity_sha,
        "preflight_sha256": hashlib.sha256(raw).hexdigest(),
        "force_audit_sha256": preflight["force_audit_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_stage", type=Path)
    parser.add_argument("validation_stage", type=Path)
    parser.add_argument("normaliser", type=Path)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new attempt directory")
    parser.add_argument("--resume", type=Path, help="resume.pt from a prior attempt")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    device = full_a100()
    torch.set_float32_matmul_precision("highest")

    train_rows, val_rows, normaliser_bytes, inputs = training_inputs(
        args.train_stage, args.validation_stage, args.normaliser
    )
    config = OriginalTrainConfig()
    inputs["qualification"] = check_training_qualification(
        args.qualification,
        inputs,
        normaliser_bytes,
        config,
        {
            "gpu": torch.cuda.get_device_name(device),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
        },
    )
    norm = Normaliser.from_dict(json.loads(normaliser_bytes))

    training = pack_cases(args.train_stage, train_rows, norm)
    validation = pack_cases(args.validation_stage, val_rows, norm)
    print(
        f"Loaded {len(training)} training and {len(validation)} validation cases",
        flush=True,
    )
    print(
        train_original(
            config,
            training,
            validation,
            device=device,
            out=args.out,
            normaliser_bytes=normaliser_bytes,
            inputs=inputs,
            resume=args.resume,
        )
    )


if __name__ == "__main__":
    main()
