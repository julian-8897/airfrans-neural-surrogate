"""Account for the proposed one-A100 campaign from Slurm and exp records."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from airfoil_surrogate.cfd.accounting import LIVE, TERMINAL

GPU_CAP_SECONDS = 24 * 3600
TRAINING_MAX_SECONDS = 23 * 3600
PHASE_LIMIT_MINUTES = {
    "qualification": 30,
    "training": TRAINING_MAX_SECONDS // 60,
    "official": 30,
    "transfer": 30,
}
INFERENCE_RESERVE_MINUTES = 30


@dataclass(frozen=True)
class GPUAllocation:
    allocated_a100: int
    elapsed_seconds: int
    limit_minutes: int
    state: str
    exit_code: str

    @property
    def charge(self) -> int:
        if self.state in LIVE:
            return max(self.elapsed_seconds, self.limit_minutes * 60)
        return self.allocated_a100 * self.elapsed_seconds

    @property
    def succeeded(self) -> bool:
        return self.state == "COMPLETED" and self.exit_code == "0:0"


def _full_a100s(tres: str) -> int:
    fields = dict(field.split("=", 1) for field in tres.split(",") if field)
    count = int(fields.get("gres/gpu:a100", 0))
    total = int(fields.get("gres/gpu", count))
    other = {key for key in fields if key.startswith("gres/gpu:")} - {"gres/gpu:a100"}
    if other or total != count or count not in (0, 1):
        raise ValueError(f"expected only a single full A100 or no allocation: {tres}")
    return count


def read_gpu_allocations(text: str) -> dict[str, GPUAllocation]:
    """Read allocation-only Slurm rows, rejecting duplicate or non-A100 jobs."""
    allocations = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        job, allocated, requested, elapsed, limit, raw_state, exit_code = (
            value.strip() for value in line.split("|")
        )
        state = raw_state.partition(" ")[0]
        if not job.isdecimal() or job in allocations or state not in LIVE | TERMINAL:
            raise ValueError(f"ambiguous GPU allocation record: {job!r}, {raw_state!r}")
        allocation = GPUAllocation(
            _full_a100s(allocated), int(elapsed), int(limit), state, exit_code
        )
        if (
            _full_a100s(requested) != 1
            or allocation.elapsed_seconds < 0
            or allocation.limit_minutes <= 0
            or (
                (allocation.elapsed_seconds > 0 or allocation.succeeded)
                and allocation.allocated_a100 != 1
            )
        ):
            raise ValueError(f"invalid full-A100 allocation or duration for {job}")
        allocations[job] = allocation
    return allocations


def read_gpu_ledger(path: Path, experiment: str) -> dict[str, dict]:
    """Bind each allocation to its exp run, including live runs before exp end."""
    ledger = json.loads(path.read_bytes())
    records, run_ids = {}, set()
    for attempt in ledger["attempts"]:
        job = str(attempt["job_id"])
        phase = attempt["phase"]
        if not job.isdecimal() or job in records or phase not in PHASE_LIMIT_MINUTES:
            raise ValueError(f"invalid or duplicate GPU attempt: {job}, {phase}")
        run_path = (path.parent / attempt["exp_run"]).resolve()
        raw = run_path.read_bytes()
        run = json.loads(raw)
        if (
            run["experiment"] != experiment
            or run["backend"]["kind"] != "slurm"
            or run["run_id"] in run_ids
        ):
            raise ValueError(f"GPU attempt requires its own Slurm exp run: {job}")
        tracked_job = run["backend"].get("job_id")
        if (
            tracked_job is not None
            and str(tracked_job) != job
            or tracked_job is None
            and run["status"] != "running"
        ):
            raise ValueError(f"GPU attempt disagrees with its exp job ID: {job}")
        run_ids.add(run["run_id"])
        records[job] = {
            "phase": phase,
            "run_id": run["run_id"],
            "status": run["status"],
            "exp_run": {
                "path": str(run_path),
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
        }
    return records


def gpu_budget(
    ledger: dict[str, dict],
    allocations: dict[str, GPUAllocation],
    *,
    phase: str | None = None,
    minutes: int = 0,
    reserve_minutes: int = 0,
    after_training: str | None = None,
) -> dict:
    """Charge terminal use and active reservations before proposing one allocation."""
    if ledger.keys() != allocations.keys():
        raise ValueError("GPU accounting snapshot and registered job IDs differ")
    if reserve_minutes < 0 or phase is None and minutes != 0:
        raise ValueError("invalid GPU reservation")
    if after_training is not None and phase != "official":
        raise ValueError("a training dependency applies only to official inference")
    if phase is not None and not 0 < minutes <= PHASE_LIMIT_MINUTES[phase]:
        raise ValueError(f"requested wall time exceeds or is outside the {phase} limit")

    succeeded, unsettled, attempts = set(), [], []
    for job, entry in ledger.items():
        allocation = allocations[job]
        candidate = entry["phase"]
        status = entry["status"]
        if allocation.limit_minutes > PHASE_LIMIT_MINUTES[candidate]:
            raise ValueError(f"job {job} exceeds its phase wall limit")
        if status != "running" and allocation.succeeded != (status == "succeeded"):
            raise ValueError(f"Slurm and exp disagree about success for job {job}")
        if status == "succeeded":
            if candidate in succeeded:
                raise ValueError(
                    f"more than one successful GPU attempt for {candidate}"
                )
            succeeded.add(candidate)
        if allocation.state in LIVE or status == "running":
            unsettled.append(job)
        attempts.append(
            {
                "job_id": job,
                **entry,
                **asdict(allocation),
                "charged_gpu_seconds": allocation.charge,
            }
        )

    if phase in ("qualification", "training"):
        reserve_minutes = max(reserve_minutes, INFERENCE_RESERVE_MINUTES)
    charged = sum(allocation.charge for allocation in allocations.values())
    proposed = minutes * 60
    reserved = reserve_minutes * 60
    within_cap = charged + proposed + reserved <= GPU_CAP_SECONDS
    prerequisite = {
        "qualification": None,
        "training": "qualification",
        "official": "training",
        "transfer": "training",
    }.get(phase)
    serial_dependency = (
        after_training is not None
        and unsettled == [after_training]
        and ledger[after_training]["phase"] == "training"
        and ledger[after_training]["status"] == "running"
        and allocations[after_training].state in LIVE
        and "qualification" in succeeded
    )
    return {
        "charged_gpu_seconds": charged,
        "proposed_gpu_seconds": proposed,
        "reserved_future_gpu_seconds": reserved,
        "cap_gpu_seconds": GPU_CAP_SECONDS,
        "within_cap": within_cap,
        "phase": phase,
        "after_training": after_training,
        "unsettled_jobs": unsettled,
        "successful_phases": sorted(succeeded),
        "can_submit": (
            phase is not None
            and within_cap
            and (serial_dependency if after_training is not None else not unsettled)
            and phase not in succeeded
            and (prerequisite is None or prerequisite in succeeded or serial_dependency)
        ),
        "attempts": attempts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("ledger", type=Path)
    parser.add_argument("accounting", type=Path)
    parser.add_argument("--phase", choices=PHASE_LIMIT_MINUTES)
    parser.add_argument("--minutes", type=int, default=0)
    parser.add_argument("--reserve-minutes", type=int, default=0)
    parser.add_argument("--after-training", help="active training job dependency")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    accounting = args.accounting.read_bytes()
    result = gpu_budget(
        read_gpu_ledger(args.ledger, args.experiment),
        read_gpu_allocations(accounting.decode()),
        phase=args.phase,
        minutes=args.minutes,
        reserve_minutes=args.reserve_minutes,
        after_training=args.after_training,
    )
    with args.out.open("x") as stream:
        json.dump(
            result
            | {
                "experiment": args.experiment,
                "accounting": {
                    "path": str(args.accounting.resolve()),
                    "sha256": hashlib.sha256(accounting).hexdigest(),
                },
            },
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
    if not result["within_cap"] or args.phase and not result["can_submit"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
