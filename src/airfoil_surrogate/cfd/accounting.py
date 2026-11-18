"""Account for CFD allocations from an allocation-only Slurm snapshot."""

from __future__ import annotations

from dataclasses import dataclass

TOTAL_CORE_SECONDS = 600 * 3600
WALL_LIMIT_MINUTES = {"reproduction": 50, "matched": 35}
CORES = 16
LIVE = {
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "RESIZING",
    "SUSPENDED",
    "SIGNALING",
    "STAGE_OUT",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
}
TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "BOOT_FAIL",
    "PREEMPTED",
    "OUT_OF_MEMORY",
    "DEADLINE",
    "REVOKED",
}


@dataclass(frozen=True)
class Allocation:
    allocated_cpus: int
    requested_cpus: int
    elapsed_seconds: int
    limit_minutes: int
    state: str
    exit_code: str

    @property
    def charge(self) -> int:
        if self.state in LIVE:
            return max(self.allocated_cpus, self.requested_cpus) * max(
                self.elapsed_seconds, self.limit_minutes * 60
            )
        return self.allocated_cpus * self.elapsed_seconds


def read_allocations(text: str) -> dict[str, Allocation]:
    """Read sacct -X -n -P with the fields documented for the CFD campaign."""
    records = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        job, allocated, requested, elapsed, limit, raw_state, exit_code = (
            value.strip() for value in line.split("|")
        )
        state = raw_state.split()[0]
        if not job.isdecimal() or job in records or state not in LIVE | TERMINAL:
            raise ValueError(f"ambiguous allocation record: {job!r}, {raw_state!r}")
        a = Allocation(
            int(allocated), int(requested), int(elapsed), int(limit), state, exit_code
        )
        if (
            not 0 <= a.allocated_cpus
            or not 0 <= a.elapsed_seconds
            or a.requested_cpus != CORES
            or a.limit_minutes <= 0
            or (a.elapsed_seconds and not a.allocated_cpus)
        ):
            raise ValueError(f"invalid CPU allocation or elapsed time for {job}")
        records[job] = a
    return records


def budget(
    allocations: dict[str, Allocation],
    phases: dict[str, str],
    proposals: dict[str, int],
) -> dict:
    if allocations.keys() != phases.keys():
        raise ValueError("accounting snapshot and registered job IDs differ")
    for job, phase in phases.items():
        if allocations[job].limit_minutes > WALL_LIMIT_MINUTES[phase]:
            raise ValueError(f"{job} exceeds its phase's wall-time limit")
    if any(count < 0 for count in proposals.values()):
        raise ValueError("proposed job count must be non-negative")
    charged = sum(a.charge for a in allocations.values())
    proposed = sum(
        count * CORES * WALL_LIMIT_MINUTES[phase] * 60
        for phase, count in proposals.items()
    )
    return {
        "charged_core_seconds": charged,
        "proposed_core_seconds": proposed,
        "cap_core_seconds": TOTAL_CORE_SECONDS,
        "within_cap": charged + proposed <= TOTAL_CORE_SECONDS,
    }
