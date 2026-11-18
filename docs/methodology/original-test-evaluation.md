# Evaluate the frozen Transolver on the official full-task test set

After the [checkpoint and baselines are frozen](original-training.md#fit-the-baselines-and-freeze-the-evaluation-inputs), stage the complete official test partition against the same freeze, then run the retained evaluation in its registered job:

```bash
uv run --locked python -m airfoil_surrogate.data.stage_original /path/to/Dataset.zip \
  --partition test --frozen /path/to/frozen.json --out /path/to/new/test

python -m airfoil_surrogate.eval.original run \
  /path/to/completed/training /path/to/baselines.json /path/to/frozen.json \
  /path/to/test --out "$EXP_OUT/test"
```

The evaluator checks all frozen file hashes, then verifies the test stage and reads the meshes in official order. Transolver sees each complete native point set once. The prediction `.npz` contains dimensional fields in original node order. Each row of `cases.jsonl` records the native mesh digests and prediction digest. Field errors use the raw prediction. Primary lift and drag use exact no-slip on the predicted wall velocity, and the raw-field coefficients are retained separately.

`summary.json` contains equal-case summaries, independent lift and drag Spearman correlations, and paired model-minus-baseline mean errors with 95% percentile intervals. The bootstrap draws 10,000 paired case resamples with seed 0. A zero reference norm or coefficient gives an undefined relative error, and the summaries record the defined counts. Regenerate the summary from the complete saved case metrics without running inference:

```bash
python -m airfoil_surrogate.eval.original summarise /path/to/evaluation-output \
  --out /path/to/new/summary.json
```
