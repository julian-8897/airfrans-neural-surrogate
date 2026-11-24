# Final UIUC timeout recovery

- Date: 2026-11-24
- Status: approved by the user
- PLAN.md item: Compute budget

Raise the cumulative cap to 725 allocated CPU-core-hours and allow one 50-minute, 16-core retry of `uiuc-s3016` after [Slurm timeout 928510](../../experiments/2026-11-23-airfrans-frozen-geometry-transfer/runs/20261124T121440Z-8e65/run.json). The preceding attempt reached iteration 19,938 of 20,000 and saved no final flow fields. All 59 other CFD references have qualified. The case ranks, operating point, solver, CFD checks, frozen model and 24 A100-hour cap follow the [existing protocol](../methodology/original-mesh-transfer-protocol.md).

The settled spend at approval was 705.08 CPU-core-hours. Reserve 13⅓ CPU-core-hours for the retry and two for freeze, inference and verification. Use `--cpu-core-hours 725 --matched-timeout-recovery --final-matched-recovery` alongside `--reference-requalification` for final accounting.
