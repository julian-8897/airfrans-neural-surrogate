# Geometry-aware airfoil surrogates: plan

Status: draft, 2026-11-17 (scoping checks added the same day). Decisions marked **[agree]** are plan-level and must be settled before the first reported run; each gets a dated record in `docs/decisions/`.

## Purpose

A portfolio project in research-production style: a mesh-native surrogate for 2D RANS airfoil flow, trained and benchmarked on AirfRANS, then tested on the two things industrial users care about: shapes outside the training family, and whether design improvements found with the surrogate survive a real CFD check. It ships as a containerised service with a reproducible evaluation harness.

This is stage 1. Stage 2 (later, separate plan) ports the pipeline and harness to 3D car aerodynamics (AhmedML) with multi-GPU training.

## Research questions

- **Q1 Benchmark.** On the four AirfRANS tasks (full, scarce, Reynolds extrapolation, angle-of-attack extrapolation), how do mesh-native models compare with a parametric baseline on fields and on the lift and drag coefficients (CL, CD)?
- **Q2 Out-of-family geometry.** On airfoils outside the NACA family, how much does mesh-model error grow relative to the NACA test set? (The parametric baseline cannot be applied here at all.)
- **Q3 Design loop.** When an optimiser improves a design using the surrogate, what fraction of the predicted gain survives OpenFOAM verification? Does an ensemble-uncertainty penalty raise that fraction?

Known going in (AirfRANS paper, Tables 3 and 17): baseline field surrogates rank lift well (Spearman ρ_L ≈ 0.97) and drag badly (ρ_D between about −0.3 and 0.1), because wall shear stress is poorly predicted. Any objective that includes drag is therefore a test of trust, not a safe target.

A negative answer to Q2 or Q3 is an acceptable result if the evidence is clean.

## Scope

In scope: 2D incompressible steady RANS (AirfRANS setup), mesh-native models, ensembles for uncertainty, OpenFOAM reproduction of the AirfRANS case setup, a design loop with verification, and a FastAPI service.

Out of scope for stage 1: 3D, new architectures, differentiating through the mesher (CMA-ES is the optimiser; gradients are a stretch goal), Tesseract, LLM agents.

## Data

- **AirfRANS:** 1000 simulations of NACA 4- and 5-digit airfoils, Re 2 to 6 × 10⁶, angle of attack −5° to 15°. Standard split 800/200; use the dataset's four task splits unchanged. Install with `pip install airfrans`.
- **Licence:** the Extrality/AirfRANS repo LICENSE is ODbL 1.0 (checked 2026-11-17). ODbL is share-alike for derivative databases, so converted shards and any published derived data must stay ODbL with attribution. Whether trained weights count as a derivative database or a Produced Work is unresolved; do not publish weights until settled. (HAL lists CC BY-NC-SA, probably for the paper only.)
- **Preprocessing:** convert to a sharded on-disk format (zarr) with per-file checksums and a manifest. Normalisation statistics come from the training split only.
- **Leakage check:** confirm that no airfoil shape appears in both train and test under different flow conditions. If shapes repeat, add a shape-held-out split. **[agree]**
- **Out-of-family set:** 30 to 50 non-NACA airfoils (UIUC database), simulated with the reproduced OpenFOAM setup at conditions inside the training range. Selection rule fixed before any model sees them. **[agree]**

## Solver reproduction

Needed for Q2 and Q3. Rebuild the AirfRANS OpenFOAM case (mesh, boundary conditions, `simpleFoam` with k-ω SST) and re-run about 10 AirfRANS cases.

Gate: CL and CD within a tolerance of the dataset values, and surface pressure visually matching, before any new simulation is trusted. **[agree tolerance]**

Checked 2026-11-17: the Extrality/AirfRANS repo ships no OpenFOAM case or meshing scripts, only training code, models and `naca_generator.py`. The case setup (mesh topology and resolution, boundary conditions, solver settings) must be rebuilt from the paper's description. One weekend is likely optimistic; budget two and treat the reproduction gate as the go/no-go for Q2 and Q3.

Cost reference: about 25 min per case on 16 CPU cores (paper, Table 4).

## Models

- **Parametric baseline:** GP or small MLP from (NACA parameters, Re, AoA) to (CL, CD). Coefficients only, NACA only.
- **GraphSAGE:** the paper's baseline. Reproducing its published numbers within the reported ± is the harness sanity check.
- **Transolver:** the main mesh-native model. Use the official implementation if the licence allows. Compare with Transolver's published AirfRANS numbers.
- **Ensembles:** 5 seeds per mesh model, used both for mean ± std reporting and as the uncertainty signal.

Architectures and hyperparameters for reported runs: **[agree]**. The default is published configs, with no tuning on the test split.

Inputs follow AirfRANS: node position, inlet velocity, signed distance and normals. Outputs: u_x, u_y, p, ν_t.

## Evaluation harness

Built before any model work, and tested on ground truth first.

- **Field errors:** MSE on normalised fields (volume and surface, matching the paper), plus per-case relative L2.
- **Coefficients:** relative error and Spearman correlation for CL and CD, computed with the same force integration as the paper. Use the `airfrans` library's `Simulation` force-coefficient routine rather than reimplementing it, and cross-check it in the invariant tests.
- **Tails:** per-case P95 and P99 errors, not only means.
- **Boundary layer diagnostic:** wall shear stress error, to track the drag failure mode.
- **Invariant tests:** ground-truth fields pushed through the force integration reproduce the dataset coefficients; uniform pressure gives zero net force; fixed seeds give identical outputs.
- **Reporting:** mean ± std over 5 seeds, every number traceable to an `exp` run.

Metric set and which runs count: **[agree]**

## Design loop (Q3)

- **Design variables:** NACA 4-digit parameters (max camber, camber position, thickness) at fixed Re and AoA inside the training range.
- **Objectives (two arms):** **[agree]**
  - A, lift-led control: maximise CL subject to a minimum thickness and to bounds kept inside the training range. Expected to transfer well.
  - B, realistic: maximise CL/CD with the same constraints. Expected to expose the drag weakness.
- **Optimiser:** CMA-ES on the surrogate. Each candidate is meshed and featurised, so the mesher sits inside the loop.
- **Uncertainty penalty:** score = μ − κσ from the ensemble, κ ∈ {0, κ*}, with κ* fixed on validation data before verification.
- **Arms:** {parametric baseline, Transolver ensemble} × {κ = 0, κ*} × {A, B}, top 10 designs each, giving 80 verification runs.
- **Outputs:**
  - realised against predicted improvement over the starting design
  - fraction of designs with a real gain
  - error at the optimum compared with test-set error, which shows the optimiser exploiting surrogate error

## Service

- FastAPI in Docker. Input: airfoil coordinates (Selig `.dat`), Re and AoA. Output: CL, CD, ensemble σ, an out-of-distribution flag, and optionally downsampled fields.
- The OOD flag thresholds ensemble σ, calibrated on the test split, and is reported with its coverage against error.
- Report latency p50 and p95 per request including meshing, next to the 25-minute CFD wall time.
- CI: unit and invariant tests, golden-output regression tests on 3 fixed airfoils, and a container build.
- Model card: training range, known failure modes (drag, out-of-range Re and AoA), and the OOD flag's measured coverage.

## Compute budget **[agree]**

- **GPU:** 2 models × 5 seeds × 4 tasks, a few hours each on one modern GPU (paper baselines took 2 to 7 h on an RTX 3090). Estimate before launching; Transolver cost unverified.
- **CPU (OpenFOAM):** about 10 reproduction + 30 to 50 out-of-family + 80 verification ≈ 120 to 140 cases × 25 min × 16 cores, so roughly 50 to 60 node-hours on HPC.

## Milestones

| | Milestone | Exit criterion |
|---|---|---|
| M0 | Repo, `uv`, data download, licence check, `exp` experiment registered | data checksums recorded; licence noted |
| M1 | Data pipeline and evaluation harness | invariant tests pass on ground truth; GraphSAGE reproduces paper numbers within ± |
| M2 | Parametric baseline and Transolver ensembles on 4 tasks | Q1 table with 5 seeds |
| M3 | OpenFOAM reproduction and out-of-family set | reproduction gate passed; Q2 table |
| M4 | Design loop and verification | Q3 figure and table |
| M5 | Service, CI, model card, README write-up | container serves; latency measured; results in README |

Rough effort: 4 to 5 weekends. If time runs short, cut objective B and the 5-digit family first. Keep M1, M3 and the lift arm of M4, since the out-of-family test and verification are what make the project distinctive.

## Risks

- **Case-generation scripts missing (confirmed):** rebuilding the OpenFOAM setup delays M3 and M4. Proposed fallback, not yet agreed: if the reproduction gate cannot be passed, Q2 and Q3 report results against a documented, self-consistent setup, which weakens their claims.
- **Parametric baseline wins on in-family coefficients:** report it. The geometry argument then rests on Q2.
- **Drag remains unpredictable:** objective B becomes a documented negative result, and the uncertainty gate is the deliverable.
- **Meshing inside the loop is slow or fragile:** fall back to a fixed query point cloud built from the parametric surface.
- **Licence forbids redistributing weights:** publish code and results only.

## Proposed repo layout

```
AGENTS.md            # single source of truth for agents (CLAUDE.md imports it)
PLAN.md              # this file
docs/decisions/      # dated decision records for each [agree] item
src/airfoil_surrogate/
  data/              # AirfRANS conversion, splits, featurisation, meshing
  models/            # parametric, graphsage, transolver, ensemble wrapper
  eval/              # metrics, force integration, reports
  design/            # CMA-ES loop, penalty, candidate export
  service/           # FastAPI app
cfd/                 # OpenFOAM case templates and run scripts (SLURM)
configs/
tests/
```

## References

- Bonnet et al. 2022, AirfRANS, NeurIPS Datasets and Benchmarks, arXiv:2212.07564
- Wu et al. 2024, Transolver, ICML; code: github.com/thuml/Transolver
- Ashton et al. 2024, WindsorML / AhmedML / DrivAerML (stage 2), arXiv:2407.19320
