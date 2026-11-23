# Recover matched CFD cases that exceed their first allocation

- Date: 2026-11-23
- Status: approved by the user
- PLAN.md item: Compute budget

Raise the [cumulative matched transfer cap](2026-11-23-matched-cpu-cap.md) to 710 allocated CPU-core-hours. First attempts keep the 35-minute wall limit. Same-input retries after a recorded Slurm timeout may request up to 45 minutes on 16 cores. All allocations, including failures and live reservations, count towards the cap. The 24 A100-hour cap and the separate reference requalification cap remain in force.

At the time of the approval request, all 30 NACA controls and 19 UIUC cases had qualified. Five UIUC candidates had unresolved timeouts, and a sixth timed out while new candidates were running. The earlier 35-minute retries on different nodes showed that scheduling alone did not reliably give these cases enough time. The extra wall time allows the existing iteration schedule and export to finish. The original cohort ranking, convergence, mesh, physical conditions and inference rules still apply. [Matched allocation records](../../experiments/2026-11-23-airfrans-frozen-geometry-transfer/experiment.json)
