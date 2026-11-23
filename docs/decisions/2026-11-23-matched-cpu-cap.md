# Matched transfer CPU allocation cap

- Date: 2026-11-23
- Status: approved by the user
- PLAN.md item: Compute budget

Raise the cumulative CPU cap for the original-mesh benchmark, reference requalification and matched transfer to 675 allocated CPU-core-hours. The [600-hour cap](2026-11-20-original-transfer-budget.md) applied to allocations submitted before this approval. Keep the 24 A100-hour cap and the [separate 50 CPU-core-hour requalification cap](2026-11-23-reference-mesh-requalification.md).

At 26 qualified matched cases, the settled allocations had used 336.03 CPU-core-hours, including the original benchmark, requalification, mesh rejects and timeouts. At the observed mean of 8.01 CPU-core-hours per successful case, the remaining 34 successes alone projected to 608.42 cumulative CPU-core-hours. The revised total leaves room for the existing ranked replacements, same-input recovery and report verification. The [registered matched CFD attempts](../../experiments/2026-11-23-airfrans-frozen-geometry-transfer/experiment.json) provide the underlying allocation records.

Use the original case ranks, operating point, 16-core and 35-minute matched jobs, mesh and force convergence checks, and frozen Transolver. Account for all previous allocations, new failed attempts, and booked resources when admitting each wave.
