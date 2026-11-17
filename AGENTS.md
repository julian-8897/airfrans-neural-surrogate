# AGENTS.md: airfrans-geometry-neural-surrogate

Single source of truth for agents working in this repo. `CLAUDE.md` imports this file; do not add a separate one. The research plan, with methods, budgets and milestones, is in [`PLAN.md`](PLAN.md). This file holds the rules and current state.

## Project

A mesh-native, geometry-aware surrogate for 2D incompressible steady RANS airfoil flow (AirfRANS: 1000 OpenFOAM `simpleFoam` k-ω SST cases on NACA 4- and 5-digit airfoils, Re 2 to 6 × 10⁶, AoA −5° to 15°). It is built as a research-production portfolio piece: reproducible benchmark, out-of-family geometry test, a design loop verified by CFD, and a containerised service.

- **Q1 Benchmark:** mesh-native models (GraphSAGE, Transolver ensembles) against a parametric baseline on the four AirfRANS tasks, for fields and CL/CD.
- **Q2 Out-of-family geometry:** error growth on non-NACA (UIUC) airfoils simulated with a reproduced OpenFOAM setup.
- **Q3 Design loop:** fraction of surrogate-predicted design gain that survives OpenFOAM verification, with and without an ensemble-uncertainty penalty.

Stage 2 (3D car aerodynamics, AhmedML, multi-GPU) is a separate, later plan.

## Current state

- 2026-11-17: repo scaffolded (uv skeleton, plan, licence, pre-commit).
- 2026-11-17: **prototyping on `PLAID-datasets/AirfRANS_remeshed`**; see [the decision record](docs/decisions/2026-11-17-prototype-data-remeshed.md). The data comes from `scripts/download_remeshed.py`, with revisions and SHA-256 checksums pinned in `manifests/airfrans_remeshed.json`, and lands in `data/raw/airfrans_remeshed/`. The loader is `airfoil_surrogate.data.remeshed`. Remeshed results are **not comparable** with published AirfRANS or Transolver numbers; the data for reported runs is still open.
- Next: featurisation (AirfRANS inputs: position, inlet velocity, signed distance, normals) and the harness on remeshed data; checking the stored CL/CD against the `airfrans` force integration; registering the `exp` experiment once the question, protocol and budget are agreed.

## Repo rules

- **Plan-level decisions:** every **[agree]** item in PLAN.md gets a dated record in `docs/decisions/` before the first reported run that depends on it. The index is in [`docs/decisions/README.md`](docs/decisions/README.md). Do not settle an [agree] item without the user.
- **Tracking:** the tracker is `exp` (experiment-tracking skill). Every reported number, table and figure traces to an `exp` run.
- **Data is read-only:** never modify or overwrite raw AirfRANS files, converted shards, checkpoints or simulation outputs that the current task did not create. Each run writes to a new directory under `runs/`.
- **Splits and leakage:**
  - Use the four AirfRANS task splits (full, scarce, reynolds, aoa) unchanged.
  - Normalisation statistics come from the training split only.
  - No tuning or model selection on test splits.
  - Out-of-family and design-verification CFD cases are never used for training or selection.
  - The selection rule for the out-of-family set is fixed before any model sees it.
- **Force coefficients:** CL and CD use the `airfrans` library's force integration, so they match the paper. Invariant tests must show that ground-truth fields reproduce the dataset coefficients.
- **Never committed:** data, runs, checkpoints and OpenFOAM outputs (see `.gitignore`). The pre-commit hook rejects files over 2 MB.

## Layout

```
AGENTS.md            # this file (CLAUDE.md imports it)
manifests/           # pinned dataset revisions and checksums
scripts/             # download and one-off utilities
PLAN.md              # research plan
docs/decisions/      # dated decision records for each [agree] item
src/airfoil_surrogate/
  data/              # AirfRANS conversion, splits, featurisation, meshing
  models/            # parametric, graphsage, transolver, ensemble wrapper
  eval/              # metrics, force integration, reports
  design/            # CMA-ES loop, penalty, candidate export
  service/           # FastAPI app
cfd/                 # OpenFOAM case templates and SLURM run scripts (cfd/cases/ is ignored)
configs/
tests/
```

Subpackages are created when their first code lands, not ahead of time.

## Compute

- The local Mac (Apple silicon) is for development, tests and small CPU checks only.
- GPU training and OpenFOAM runs happen on HPC, submitted through SLURM scripts in `cfd/` and `configs/`. Budgets are in PLAN.md (Compute budget, an [agree] item).
- Runs over about 30 minutes, any cluster job and any change to an existing job need the user's approval with a stated budget.

## Data and licence

- AirfRANS (Bonnet et al. 2022, arXiv:2212.07564) is under **ODbL 1.0**, per the Extrality/AirfRANS repo LICENSE checked on 2026-11-17. Derived databases, such as converted shards, must stay ODbL with attribution.
- This repo's code is MIT (`LICENSE`).

## Open items

- Whether trained weights are an ODbL derivative database or a Produced Work. Do not publish weights until this is settled.
- OpenFOAM case rebuild: upstream ships no meshing or case scripts, so the setup must be reconstructed from the paper. This gates Q2 and Q3.
- Transolver: the licence of the official code (github.com/thuml/Transolver) has not been checked, and its AirfRANS training cost has not been verified.
- Leakage check: done 2026-11-17. There are 1000 distinct NACA shapes and none is shared between train and test, so no extra split is proposed. Awaiting the user's agreement.
- Remeshed data: the meaning of the `zone_0..5` cell indicators is undocumented, and it is unconfirmed whether the stored CL/CD come from the original simulations.

## Verification commands

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest                      # data tests skip if data/raw/airfrans_remeshed is absent
uv run pytest -m "not slow"        # skip tests that read all 1000 samples (~15 s)
uv run pre-commit run --all-files
uv run python scripts/download_remeshed.py   # fetch and verify prototype data (0.6 GB)
```

Downloaded samples are pickles. Load them only through `airfoil_surrogate.data.remeshed.unpickle_sample`, which allows numpy globals only; never call `pickle.loads` on them directly.

## Conventions

- Python 3.12 via `uv`; add dependencies with `uv add` only when code uses them.
- Use simple functions and dataclasses. Validate inputs at file and external-data boundaries, not on internal paths.
- Numerical code gets invariant tests with fixed seeds: shapes, dtypes, finiteness, symmetry, zero net force under uniform pressure, and ground truth reproducing the dataset coefficients.
- Before refactoring code that feeds a reported number, pin its outputs with a regression test.
- British English in prose, comments, docs and commit messages.
