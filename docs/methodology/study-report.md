# Assemble the final study report

Use this command after the registered study runs finish. It reads the retained results, checks their provenance and writes a short Markdown report with a machine-readable evidence manifest. Keep the report and artefacts in private experiment outputs.

Make a JSON input file. Paths in the input file are relative to that file or absolute. Run output directories come from the `exp` records, where relative paths are relative to the project root. Use the approved protocol version and the same numerical environment as the retained evaluations so their saved summaries can be reproduced exactly.

```json
{
  "experiment": "YYYY-MM-DD-original-mesh-transfer",
  "runs": {
    "force_audit": "force-audit/run.json",
    "qualification": "a100-qualification/run.json",
    "training": "retained-training/run.json",
    "baselines": "baselines/run.json",
    "official": "official-test/run.json",
    "transfer": "matched-transfer/run.json"
  },
  "protocol": "approved-protocol.md",
  "train_stage": "stages/train",
  "validation_stage": "stages/validation",
  "test_stage": "stages/test",
  "freeze": "frozen.json",
  "cohorts": "final-cohorts.json",
  "gpu_budget": "final-a100-accounting.json"
}
```

Preserve the paths in the final cohort freeze, including the screening directory, reproduction reports, pressure overlays and registered CFD attempt records. Use the final A100 accounting snapshot with all allocations settled and zero proposed time.

Run the assembly as a tracked CPU job in the existing experiment:

```bash
exp run "$EXPERIMENT" --config "$SPEC" -- python -m airfoil_surrogate.eval.study_report \
  "$SPEC" --out "$NEW_REPORT_DIRECTORY"
```

The command checks that both evaluations used the same frozen checkpoint, compares the ordered case rows with the staged test and matched-cohort manifests, and verifies each saved prediction digest. It regenerates the summaries and all transfer rankings from the saved per-case metrics. The report links the qualification, official and matched results to their `exp` records and links the selected checkpoint, normaliser, data manifests, pressure plots and run evidence. Assembly reads no mesh fields and does no inference.

The command requires the complete study. If a qualification gate fails or a cohort falls short, retain the completed artefacts and record the failed milestone and allocation evidence in the experiment verdict and the report.

When the matched CFD gate fails and the official test completes, use
`python -m airfoil_surrogate.eval.official_report INPUTS --out REPORT_DIRECTORY`
on a CPU allocation with successful official inference as its dependency. It
rebuilds the official metrics, verifies their links to the frozen checkpoint
and case predictions, and includes the registered CFD failure and Slurm
accounting. The input maps the official study roles to their registered run
and job IDs and lists all study CPU and GPU allocations. Preserve the tracked
run records and reproduction evidence beside the scratch checkout. The
archived inputs and raw accounting snapshots record the provenance.
