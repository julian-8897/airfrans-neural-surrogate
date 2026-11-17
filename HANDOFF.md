# Last session (2026-11-17)

Original-mesh A100 hardware pilot submitted to Eureka2 as job 891545. Experiment `2026-11-17-original-a100-pilot`, run `20261117T221946Z-2bdb`, code `c58aa88`. SLURM accounting confirmed COMPLETED, exit 0:0, with 37 elapsed seconds on one A100. The request is one full A100 for 15 minutes, requeue disabled, within the approved one-hour cap.

The three original full-task training cases were prepared with `scripts/prepare_original_pilot.py`. All staged files passed SHA-256 checks on Eureka2. The cluster CUDA Python imported the archived entry point on the login node. The local test suite passed (28 tests). Timing uses 10 warm-up and 100 measured steps as described in the [pilot decision](docs/decisions/2026-11-17-original-a100-pilot.md).

Remote job directory: `/parallel_scratch/jc02423/airfrans-original-pilot-c58aa88`. Log: `slurm-891545.log`. The retrieved accounting, job log and metrics are in `outputs/2026-11-17-original-a100-pilot/20261117T221946Z-2bdb/`. The run and experiment are closed. The timed loop achieved 5.595 steps/s and used 3.57 GiB peak allocated CUDA memory. A 398-epoch run with 720 steps per epoch would take about 14.2 training-loop hours at that rate, before full-dataset I/O, validation and checkpointing. No jobs remain in flight. Full training and unfamiliar-geometry inference are pending.
