# Prototype validation on AirfRANS_remeshed

Run the evaluator against a saved Transolver checkpoint. It uses the validation indices and normaliser recorded by training. The task's held-out test cases remain unopened.

```bash
uv run --locked --extra forces python -m airfoil_surrogate.eval \
  runs/20261117T201901Z-smoke-full-s0 --forces
```

Omit `--extra forces` and `--forces` for field metrics alone. The evaluator defaults to CPU and writes `cases.jsonl` and `summary.json` under a new directory in `runs/`. Use `--out` to specify a new output directory or `--limit N` for a smoke check on the first N validation cases.

## Metrics

- MSE averages over nodes in each region and channel using the saved target normalisation. Volume comprises the off-wall nodes. The summary weights cases equally, as in the upstream AirfRANS evaluator. This prototype scores the output fields directly because the remeshed wall fields violate no-slip. [Upstream AirfRANS evaluator](https://github.com/Extrality/AirfRANS/blob/main/metrics.py)
- Relative L2 divides the dimensional error norm by the dimensional reference norm within each region and channel. The summary includes the mean, P95 and P99 across cases and the number of defined values. A zero reference norm gives a JSON null in that case's record.
- With `--forces`, `airfrans.Simulation.force_coefficient` integrates wall pressure and viscous traction from the near-wall velocity gradients. The adapter uses the triangulation and topological wall mask and the package's default viscosity. Per-case rows retain total, pressure and viscous contributions for CL and CD. The prototype reports relative error and Spearman correlation versus both integrated remeshed ground truth and the stored scalars. A zero reference coefficient gives undefined relative error. Spearman correlation is undefined for single-case or constant inputs. Wall shear stress RMSE is component-wise in kinematic units (m²/s²) against the remeshed reference. [AirfRANS Simulation](https://airfrans.readthedocs.io/en/latest/_modules/airfrans/simulation.html)

## Ground-truth force check

The check used `full_train` indices 0, 194 and 437. Their original VTU/VTP pairs were matched by case name to the pinned original mirror revision recorded in `manifests/airfrans_remeshed.json`. Run with those files in the standard `Dataset/<case>/<case>_{internal.vtu,aerofoil.vtp}` layout.

```bash
uv run --locked --extra forces python scripts/check_force_labels.py \
  /path/to/Dataset 0 194 437
```

Across these cases, original-mesh integration differed from the stored CL by at most 3.2 × 10⁻⁶ and from the stored CD by at most 9.0 × 10⁻⁷. That supports an original-mesh origin for the labels on this sample. Remeshed ground-truth integration yielded 33–69% lower CD than the stored values. The difference came primarily from the smaller viscous term. The original-mesh check and the analytic uniform-pressure, linear-pressure and linear-shear tests exercise the integration. The three-case check leaves dataset-wide label provenance and the reported-run force tolerance open.
