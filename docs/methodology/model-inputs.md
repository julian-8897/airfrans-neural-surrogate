# Model inputs and targets (prototype, AirfRANS_remeshed)

Status: examined 2026-11-17 on the `full_train` split (800 cases, 6.08M nodes). Reproduce with `uv run python scripts/examine_inputs.py`, which writes figures to `outputs/inputs_examination/` (not tracked).

## Channels

Code: `airfoil_surrogate.data.features`.

The channels follow the AirfRANS and Transolver convention.

| Input | Definition |
|---|---|
| `x`, `y` | node position (chord = 1, leading edge at the origin) |
| `u_inf_x`, `u_inf_y` | U∞ (cos α, sin α), constant per case |
| `sdf` | −`implicit_distance`, positive in the fluid, ≈ 0 on the wall |
| `n_x`, `n_y` | unit wall normal pointing into the fluid; zero off the wall |

Targets: `u_x`, `u_y`, `p` (kinematic, m²/s²), `nu_t`. Normalisation is a per-channel mean and standard deviation pooled over all training nodes, fitted on the training cases only.

## Findings

- **Wall nodes come from mesh topology, not from u = 0.** Wall nodes are the nodes on boundary edges that are not on the clip box. The reference code instead marks nodes with u_x = 0, but that test fails on remeshed data: only about 4% of wall nodes have u exactly 0. Every case has exactly one closed wall loop, and wall nodes make up 14.4% of all nodes, because the surface is finely resolved and the volume is coarsened.
- **The remeshed fields violate no-slip.** On wall nodes, |u|/U∞ has median 0.0064 and 99th percentile 0.046, with a maximum of 1.27, at leading-edge nodes where interpolation from the original mesh is poor. The data is left unmodified, but:
  - surface MSE on u_x and u_y partly measures interpolation error;
  - wall shear stress cannot be estimated reliably from this data.
- **Normals check out.** Each wall normal agrees in direction with the gradient of the signed distance: the 1st-percentile cosine is 1.000 and the minimum is 0.075, at trailing-edge corners. One node in one case was skipped because its surrounding triangles all lie on the wall.
- **Heavy tails.** Skewness of the normalised targets is u_x 0.13, u_y 4.45, p −5.42 and nu_t 4.69. Pressure reaches −57σ at leading-edge suction peaks. Under mean/std scaling, a handful of nodes dominate the MSE for p.
- **Scaling with U∞:**

  | Target | log-log slope of per-case RMS vs U∞ | Dimensional analysis | Case-to-case CV, raw → after dividing by U∞ⁿ |
  |---|---|---|---|
  | \|u\| | 0.99 | 1 | 0.29 → 0.06 |
  | p | 2.00 | 2 | 0.86 → 0.59 |
  | nu_t | 0.78 | 1 | 0.77 → 0.75 |

  Dividing velocity by U∞ and pressure by U∞² removes the Reynolds-number part of the case-to-case scale variation. What remains for p is mostly driven by angle of attack.

## Open choices (plan-level for reported runs)

- **Non-dimensional targets:** u/U∞ and p/U∞², fitted before mean/std scaling. The evidence above suggests this should help, especially on the Reynolds extrapolation task, but it departs from the AirfRANS convention that published numbers use. Proposed as an ablation once the baseline harness exists; not adopted.
- **Treatment of the heavy pressure tails** (loss weighting, or a robust loss). Not proposed until the harness measures per-case tails.
- **Enforcing u = 0 on wall nodes in the targets.** This would change the data. Not proposed for the prototype.
