# Prototype on AirfRANS_remeshed

- Date: 2026-11-17
- Status: accepted (prototype scope only)
- PLAN.md item: Data; Architectures and hyperparameters for reported runs (related)

## Decision

Prototype the data pipeline, harness and first models on `PLAID-datasets/AirfRANS_remeshed` at revision `10b38a5a`, on the local Mac. Which data feeds **reported** runs (Q1 to Q3 tables, the GraphSAGE reproduction check) is not decided here and stays open.

## Options considered

- **AirfRANS_remeshed (0.61 GB):** fits on the Mac and includes all 1000 cases with the official splits. Its meshes are about 24 times coarser in the volume, so its numbers are not comparable with published AirfRANS or Transolver results.
- **Official processed dataset (10.0 GB zip) or the `OneScience-Group/airfrans` mirror (15.0 GB):** comparable with the paper, but heavy for local iteration. Better kept on the cluster.
- **A hand-picked subset of original cases:** a single code path with reported runs, but fewer cases and no full splits locally.

## Rationale

The user chose to prototype on the remeshed set. It exercises the whole pipeline (splits, featurisation, training, CL/CD error, and design-loop plumbing) on all 1000 cases at low cost, and the loader needs no extra dependencies beyond numpy, pyarrow and pyyaml.

## Evidence (checked 2026-11-17)

Scripts: `scripts/download_remeshed.py` (pinned revisions, SHA-256 verified), `src/airfoil_surrogate/data/remeshed.py`, `tests/test_remeshed.py`.

- **Integrity:** local SHA-256 checksums of the three Parquet shards match the upstream LFS checksums for the pinned revision; they are recorded in `manifests/airfrans_remeshed.json`.
- **Safety:** each sample is a pickle. A numpy-only allow-list unpickler loads all 1000 samples, and the only globals referenced are numpy `ndarray`, `dtype`, `_reconstruct` and `scalar`.
- **Contents:**
  - Triangular mesh with vertex fields `Ux`, `Uy`, `p`, `nut` and `implicit_distance` (negative in the fluid).
  - Six cell-centred `zone_k` indicators that partition the cells. What the zones mean is not documented; they are possibly the MMG remeshing regions.
  - Scalars `C_L`, `C_D`, `inlet_velocity` (m/s) and `angle_of_attack` (radians).
  - Connectivity is 1-based (CGNS); the loader converts it to 0-based.
- **Ranges:** U∞ 31.3 to 93.6 m/s (Re 2.0 to 6.0 × 10⁶, with ν = 1.56 × 10⁻⁵ and chord 1); AoA −4.9° to 14.9°; CL −0.53 to 1.89; CD 0.0069 to 0.046.
- **Resolution:**
  - Volume: 4.2k to 9.7k nodes per case (median 7.7k). The original case `airFoil2D_SST_31.283_-4.156_0.919_6.98_14.32` has 177k nodes and 176k quad cells, against 7.4k nodes for the same case remeshed: about 24 times coarser.
  - Airfoil surface: 473 to 1670 wall nodes (median 1116). Case 0 has 1202, against 994 surface points in the original, so MMG kept or refined the wall and coarsened the volume.
  - Near-wall spacing: the median closest off-wall node is at |d| ≈ 3.4 × 10⁻⁵ chords. I have not compared this with the original first-cell height.
- **Domain:** clipped to [−2, 4] × [−1.5, 1.5]. The official processed data is already cropped to about [−2.16, 4.23] × [−1.62, 1.62] (measured on the case above), so the clip removes only a thin margin.
- **Splits:** the card's index lists give the AirfRANS sizes (full 800/200, scarce 200, reynolds 504/496, aoa 804/196). Train and test are disjoint in every task, and scarce is a subset of full train.
- **Case names:** each sample maps to exactly one original case name through (U∞ to 3 d.p., AoA in degrees to 3 d.p.). The mapped splits equal the official `manifest.json` splits exactly. The manifest comes from the `OneScience-Group/airfrans` mirror at revision `f99b3778`. This recovers the NACA parameters, which the parametric baseline needs.
- **Leakage check (PLAN.md Data item):** the 1000 cases have 1000 distinct NACA parameter tuples (489 four-digit, 511 five-digit), and no shape is shared between train and test in the full, reynolds or aoa tasks. Based on this, no extra shape-held-out split appears necessary. This still needs the user's confirmation; see the index.

## Consequences and caveats

- Nothing computed on this dataset may be reported against published AirfRANS or Transolver numbers.
- The stored `C_L` and `C_D` are probably from the original simulations, but the card does not say. Before CL/CD errors on this set mean anything, check them against the `airfrans` force integration on a few original cases.
- The coarse near-wall mesh is likely to degrade wall-shear-stress and drag diagnostics more than lift. Drag results from prototyping are indicative only.
- The pipeline should keep the dataset behind one interface (a sample of positions, cells, fields, wall nodes and scalars) so the reported-data loader can replace it.

## What would change it

If the stored CL and CD turn out not to match the original simulations, or the coarse mesh makes prototyping results misleading for model choice, move local work to a subset of original cases.
