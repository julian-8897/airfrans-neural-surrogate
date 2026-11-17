"""Train a field surrogate on AirfRANS_remeshed (prototype).

Follows the Transolver AirfRANS recipe: Adam with a one-cycle schedule
(max lr 1e-3), batch size 1, loss = volume MSE + w * surface MSE on normalised
fields, validation on the last 10% of the task's training list. The test split
is never loaded. Deviation: the full remeshed mesh (<10k nodes) is used every
step instead of 32k randomly subsampled nodes from the original ~180k-node mesh.

    uv run python -m airfoil_surrogate.train --task full --epochs 398
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from airfoil_surrogate.data.features import TARGET_NAMES, Normaliser, build_features
from airfoil_surrogate.data.remeshed import iter_samples, load_split_indices
from airfoil_surrogate.models.transolver import Transolver, TransolverConfig

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TrainConfig:
    task: str = "full"  # full | scarce | reynolds | aoa
    epochs: int = 398
    lr: float = 1e-3
    surf_weight: float = 1.0
    val_fraction: float = 0.1
    val_every: int = 10
    max_train_cases: int | None = None  # truncate training list (smoke runs)
    seed: int = 0
    device: str = "auto"
    data_dir: str = "data/raw/airfrans_remeshed"
    out_root: str = "runs"
    name: str = "transolver"
    model: TransolverConfig = field(default_factory=TransolverConfig)


@dataclass
class Case:
    index: int
    x: torch.Tensor  # (1, N, 7) normalised inputs
    pos: torch.Tensor  # (1, N, 2) raw positions
    y: torch.Tensor  # (1, N, 4) normalised targets
    surf: torch.Tensor  # (N,) bool


def split_train_val(cfg: TrainConfig) -> tuple[list[int], list[int]]:
    """Training list in card order; validation is its last ``val_fraction``."""
    splits = load_split_indices(ROOT / cfg.data_dir / "README.md")
    listed = [int(i) for i in splits[f"{cfg.task}_train"]]
    n_val = int(cfg.val_fraction * len(listed))
    train, val = listed[:-n_val], listed[-n_val:]
    if cfg.max_train_cases is not None:
        train = train[: cfg.max_train_cases]
    return train, val


def load_cases(
    cfg: TrainConfig, train: list[int], val: list[int], device: torch.device
) -> tuple[list[Case], list[Case], Normaliser]:
    feats = {
        s.index: build_features(s)
        for s in iter_samples(ROOT / cfg.data_dir, set(train) | set(val))
    }
    norm = Normaliser.fit(feats[i] for i in train)

    def to_case(i: int) -> Case:
        f = feats[i]
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)[None]  # noqa: E731
        return Case(
            i,
            t(norm.inputs(f.x)),
            t(f.pos),
            t(norm.targets(f.y)),
            torch.as_tensor(f.surf, device=device),
        )

    return [to_case(i) for i in train], [to_case(i) for i in val], norm


def field_losses(pred: torch.Tensor, case: Case) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-variable MSE over volume and surface nodes, each of shape (4,)."""
    err = (pred[0] - case.y[0]) ** 2
    return err[~case.surf].mean(0), err[case.surf].mean(0)


@torch.no_grad()
def evaluate(model: Transolver, cases: list[Case]) -> dict[str, list[float]]:
    model.eval()
    vol = torch.zeros(len(TARGET_NAMES), device=cases[0].x.device)
    surf = torch.zeros_like(vol)
    for c in cases:
        v, s = field_losses(model(c.x, c.pos), c)
        vol += v
        surf += s
    return {
        "val_vol": (vol / len(cases)).tolist(),
        "val_surf": (surf / len(cases)).tolist(),
    }


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def git_state() -> dict[str, str | bool]:
    rev = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT
        ).stdout.strip()
    )
    return {"rev": rev, "dirty": dirty}


def train(cfg: TrainConfig) -> Path:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = pick_device(cfg.device)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / cfg.out_root / f"{stamp}-{cfg.name}-{cfg.task}-s{cfg.seed}"
    out.mkdir(parents=True, exist_ok=False)

    train_ids, val_ids = split_train_val(cfg)
    t0 = time.perf_counter()
    train_cases, val_cases, norm = load_cases(cfg, train_ids, val_ids, device)
    norm.save(out / "normaliser.json")
    model = Transolver(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    meta = {
        "config": asdict(cfg),
        "git": git_state(),
        "device": str(device),
        "torch": torch.__version__,
        "n_params": n_params,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "load_seconds": time.perf_counter() - t0,
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2))
    print(
        f"{out.name}: {len(train_cases)} train / {len(val_cases)} val cases, "
        f"{n_params:,} params on {device}"
    )

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * len(train_cases)
    )
    with (out / "metrics.jsonl").open("w") as log:
        for epoch in range(cfg.epochs):
            model.train()
            t_epoch = time.perf_counter()
            vol_sum = torch.zeros(len(TARGET_NAMES), device=device)
            surf_sum = torch.zeros_like(vol_sum)
            for k in rng.permutation(len(train_cases)):
                c = train_cases[k]
                vol, surf = field_losses(model(c.x, c.pos), c)
                loss = vol.mean() + cfg.surf_weight * surf.mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                vol_sum += vol.detach()
                surf_sum += surf.detach()
            n = len(train_cases)
            row = {
                "epoch": epoch,
                "lr": sched.get_last_lr()[0],
                "train_vol": (vol_sum / n).tolist(),
                "train_surf": (surf_sum / n).tolist(),
                "seconds": time.perf_counter() - t_epoch,
            }
            if epoch % cfg.val_every == cfg.val_every - 1 or epoch == cfg.epochs - 1:
                row |= evaluate(model, val_cases)
            log.write(json.dumps(row) + "\n")
            log.flush()
            msg = (
                f"epoch {epoch:4d}  train vol {np.mean(row['train_vol']):.4f} "
                f"surf {np.mean(row['train_surf']):.4f}  {row['seconds']:.1f}s"
            )
            if "val_vol" in row:
                msg += (
                    f"  | val vol {np.mean(row['val_vol']):.4f} "
                    f"surf {np.mean(row['val_surf']):.4f}"
                )
            print(msg, flush=True)

    torch.save(
        {"model": model.state_dict(), "config": asdict(cfg.model)}, out / "model.pt"
    )
    return out


def parse_args() -> TrainConfig:
    d = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--task", default=d.task, choices=["full", "scarce", "reynolds", "aoa"]
    )
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--surf-weight", type=float, default=d.surf_weight)
    p.add_argument("--val-every", type=int, default=d.val_every)
    p.add_argument("--max-train-cases", type=int, default=None)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", default=d.device)
    p.add_argument("--name", default=d.name)
    p.add_argument("--n-hidden", type=int, default=d.model.n_hidden)
    p.add_argument("--n-layers", type=int, default=d.model.n_layers)
    p.add_argument("--slice-num", type=int, default=d.model.slice_num)
    a = p.parse_args()
    model = replace(
        d.model, n_hidden=a.n_hidden, n_layers=a.n_layers, slice_num=a.slice_num
    )
    return TrainConfig(
        task=a.task,
        epochs=a.epochs,
        lr=a.lr,
        surf_weight=a.surf_weight,
        val_every=a.val_every,
        max_train_cases=a.max_train_cases,
        seed=a.seed,
        device=a.device,
        name=a.name,
        model=model,
    )


if __name__ == "__main__":
    train(parse_args())
