"""Measure synchronised A100 training throughput on prepared original meshes."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from airfoil_surrogate.models.transolver import Transolver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    name = torch.cuda.get_device_name()
    if torch.cuda.device_count() != 1 or "A100" not in name or "MIG" in name:
        raise RuntimeError(f"expected one full A100, received {name}")
    args.out.mkdir(exist_ok=False)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    cases = []
    for i in range(3):
        with np.load(args.data / f"case{i}.npz") as data:
            cases.append(
                tuple(torch.from_numpy(data[k]) for k in ("x", "pos", "y", "surf"))
            )
    model = Transolver().cuda().train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=110)
    torch.cuda.reset_peak_memory_stats()
    for step in range(110):
        if step == 10:
            torch.cuda.synchronize()
            start = time.perf_counter()
        x, pos, y, surf = cases[step % len(cases)]
        ids = torch.from_numpy(rng.choice(len(x), 32000, replace=False))
        x, pos, y, surf = (a[ids].cuda() for a in (x, pos, y, surf))
        opt.zero_grad(set_to_none=True)
        error = (model(x[None], pos[None])[0] - y).square()
        loss = error[surf].mean() + error[~surf].mean()
        loss.backward()
        opt.step()
        sched.step()
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"non-finite loss at step {step}")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    metrics = dict(
        timed_steps=100,
        elapsed_seconds=elapsed,
        steps_per_second=100 / elapsed,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        gpu=name,
        final_loss=loss.item(),
    )
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
