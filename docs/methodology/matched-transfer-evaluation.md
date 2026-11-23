# Evaluate the frozen Transolver on the matched CFD cohorts

Use the selected checkpoint and training normaliser from the [official test freeze](original-training.md#fit-the-baselines-and-freeze-the-evaluation-inputs) and the [final CFD cohort freeze](cfd-campaign.md). Run the evaluation in its registered inference allocation.

```bash
python -m airfoil_surrogate.eval.transfer run \
  /path/to/completed/training /path/to/baselines.json /path/to/frozen.json \
  /path/to/final-cohorts.json --out "$EXP_OUT/transfer"
```

The evaluator requires 30 qualified UIUC cases and 15 controls from each NACA family at Re = 4 × 10⁶ and AoA = +4°. Before inference it checks the CFD qualification and candidate manifest hashes, the native mesh sizes and hashes, and the frozen training artefacts. Each prediction stores dimensional fields in the original order of the corresponding CFD mesh vertices. Each row in `cases.jsonl` links those fields and their scores to the native mesh, CFD job and `exp` run.

The reference coefficients come from force integration of the native CFD fields. The same integrator scores the model's fields, with exact wall no-slip for the primary coefficients. Raw-field coefficients are retained alongside them. Field errors use the raw predictions. The weighted MSEs use the frozen training scales and lump each cell's area equally among its vertices and each wall segment's length equally between its endpoints. The area average uses only off-wall nodes, with their weights renormalised. Each shape contributes equally to the group summaries.

`report/summary.json` contains means, P95, P99 and defined case counts per group, plus the UIUC minus NACA differences in mean errors and Spearman correlations. The 95% percentile intervals use 10,000 independent within-group draws with replacement from the two sets of shapes. The summaries record undefined draws, including draws with constant coefficient ranks. `report/rankings.json` records each shape's average descending ranks, predicted minus reference rank displacement and every discordant pair, separately for CL and CD. It counts predicted ties, reference ties and ties in both without treating any tied pair as discordant.

To regenerate both files from complete saved case scores, without inference:

```bash
python -m airfoil_surrogate.eval.transfer summarise /path/to/transfer \
  --out /path/to/new/report
```

For a matched study registered separately from its requalification, use `python -m airfoil_surrogate.eval.transfer_report INPUTS --out REPORT_DIR`. The inputs name the matched `experiment` and `transfer_run_id`, and supply the `training`, `baselines`, `freeze` and `cohorts` paths. The report checks that the reproduction attempts belong to the parent experiment. It verifies the frozen inputs and saved prediction digests, rebuilds the metrics and rankings, and links the evidence. On Eureka2, the preparation template runs this command in `transfer-report` mode.
