# Account for the one-A100 campaign

After protocol approval, register the study in `exp`. Use the same campaign ID for qualification, training and each retained inference job. This campaign excludes the earlier timing pilot. Give each submission or recovery attempt its own run and output directory.

During launch and monitoring, keep an allocation ledger with one entry per submitted job. Use the Slurm ID from `sbatch --parsable`. The `exp_run` path can be relative to the ledger. A live `exp start` record can omit the Slurm ID until `exp end --job` writes it. Close the run from allocation accounting once the job terminates.

```json
{
  "attempts": [
    {
      "phase": "qualification",
      "job_id": "12345",
      "exp_run": "path/to/qualification/run.json"
    }
  ]
}
```

Capture a fresh accounting snapshot for every submitted GPU attempt, including failures and active jobs. The budget verifier checks the allocated and requested typed A100 and total GPU counts. Allocation time is charged in seconds. Active jobs reserve at least their wall limit. [Slurm accounting fields](https://slurm.schedmd.com/sacct.html#SECTION_Job-Accounting-Fields).

```bash
sacct -X -n -P --duplicates --jobs "$GPU_JOB_IDS" \
  --format=JobIDRaw,AllocTRES%256,ReqTRES%256,ElapsedRaw,TimelimitRaw,State%40,ExitCode \
  > "$NEW_SNAPSHOT/sacct.psv"
```

Start with an empty ledger and empty accounting file for the qualification proposal. Before each submission, validate the proposed wall time against the new snapshot. Use `qualification`, `training`, `official` or `transfer` for the phase. The report records the `exp` run IDs and accounting digest. Only submit when the report says `can_submit: true`, with the same `--time` in `sbatch`. Propose one GPU job at a time. Preserve all snapshots when you retry a failed allocation.

```bash
python -m airfoil_surrogate.gpu_budget "$EXPERIMENT" "$LEDGER" "$NEW_SNAPSHOT/sacct.psv" \
  --phase "$PHASE" --minutes "$WALL_MINUTES" --reserve-minutes "$FUTURE_MINUTES" \
  --out "$NEW_SNAPSHOT/budget.json"
```

Qualification and training replace a smaller future reservation with 30 minutes for both evaluations. For either evaluation, explicitly retain the other evaluation's wall-time reservation if it is still pending. For recovery, choose the new training limit from the epochs left, the observed timings and the remaining allocation. A budget pass means the requested allocation fits. The completed qualification and latest checkpoint are the evidence that the proposed job has time to finish.

Official inference can be reserved behind running training with `--after-training "$TRAIN_JOB"`. The check requires that job to be the sole unsettled GPU allocation and includes both wall-time reservations in the cap. Submit the test-stage CPU job with an `afterok` dependency on training and the baseline fit. Submit inference with `afterok` dependencies on both the tested training allocation and the official test-stage job. The CPU job freezes the selected checkpoint before opening the held-out fields. A failed prerequisite cancels the dependent jobs.

| Phase | Job template | Inputs |
| --- | --- | --- |
| Qualification | [original-a100-qualification.sbatch](../../configs/original-a100-qualification.sbatch) | Pinned upstream module, training stage, validation stage, normaliser, force audit JSON and parity case ID |
| Training | [original-a100-training.sbatch](../../configs/original-a100-training.sbatch) | Training stage, validation stage, normaliser and qualification output directory, with a prior `resume.pt` for recovery |
| Official test | [original-a100-inference.sbatch](../../configs/original-a100-inference.sbatch) | `official`, completed training directory, baselines JSON, evaluation freeze and staged test directory |
| Matched transfer | [original-a100-inference.sbatch](../../configs/original-a100-inference.sbatch) | `transfer`, completed training directory, baselines JSON, evaluation freeze and final CFD cohort freeze |

Use the prepared Python environment and committed code for each submission. The batch jobs read `EXP_OUT` from the registered run and write to new child paths. The training entry point binds a passing parity and timing report to the input digests and checks that the A100 model and CUDA PyTorch build match qualification. The retained evaluations use the frozen production checkpoint. If an evaluation is interrupted, account for its consumed GPU time and preserve the partial artefacts before considering recovery.

After the final job, rerun the budget command without a proposed phase and save its result. Link the actual allocation totals, run records and outputs in the final report.
