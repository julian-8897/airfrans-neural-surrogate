# Last session (2026-11-17)

Original-mesh A100 hardware pilot submitted to Eureka2 as job 891545. Experiment `2026-11-17-original-a100-pilot`, run `20261117T221946Z-2bdb`, code `c58aa88`. The first accounting check showed PENDING (Priority), with zero elapsed time. The request is one full A100 for 15 minutes, requeue disabled, within the approved one-hour cap.

The three original full-task training cases were prepared with `scripts/prepare_original_pilot.py`. All staged files passed SHA-256 checks on Eureka2. The cluster CUDA Python imported the archived entry point on the login node. The local test suite passed (28 tests). Timing uses 10 warm-up and 100 measured steps as described in the [pilot decision](docs/decisions/2026-11-17-original-a100-pilot.md).

Remote job directory: `/parallel_scratch/jc02423/airfrans-original-pilot-c58aa88`. Log: `slurm-891545.log`. On completion, check `sacct -j 891545`, retrieve the log and `result/metrics.json` into the run's local output directory, close the run using `exp end` with the actual exit code and job ID, and record the experiment verdict. Full training and unfamiliar-geometry inference follow separate milestones.
