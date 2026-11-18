# Track allocations and freeze the CFD-qualified cohorts

After protocol approval, register every solver attempt with `exp`. Preserve the individual run records, solver directories and reproduction reports. Keep an execution ledger with one entry per allocation, in submission order. Here is a reproduction entry after verification:

```json
{
  "attempts": [
    {
      "phase": "reproduction",
      "case": "OFFICIAL_CASE_ID",
      "exp_run": "path/to/exp/run.json",
      "outcome": "qualified",
      "cfd_dir": "path/to/cfd",
      "verification": "path/to/reproduction.json"
    }
  ]
}
```

Use `phase: matched` for the matched cases. For failed attempts, set `outcome` to `retry` for a recoverable infrastructure failure or `reject` for a prespecified scientific failure. Include a `reason` and an `evidence` file. Valid scientific reasons are `geometry`, `mesh`, `nonfinite_output` and `force_convergence`. Keep active attempts in the ledger with `outcome: pending`. Resolve them before qualification or the final freeze.

Save the allocation rows for **all** submitted jobs in the campaign, including failures and active jobs. Use the raw fields below. `ElapsedRaw` is in seconds and `TimelimitRaw` is in minutes. The budget command charges actual allocated CPU time for terminal jobs and reserves the full wall limit for jobs that are active or pending. [Slurm accounting fields](https://slurm.schedmd.com/sacct.html#SECTION_Job-Accounting-Fields).

```bash
sacct -X -n -P -j "$JOB_IDS" \
  --format=JobIDRaw,AllocCPUS,ReqCPUS,ElapsedRaw,TimelimitRaw,State,ExitCode \
  > /path/to/new/sacct.psv

python -m airfoil_surrogate.cfd.campaign budget /path/to/ledger.json \
  /path/to/sacct.psv --reproduction 3 --out /path/to/new/budget.json
```

Use an empty ledger and accounting file for the first wave. Once all three reproductions pass the coefficient, pressure and convergence checks, seal their verification reports and allocation records:

```bash
python -m airfoil_surrogate.cfd.campaign qualify /path/to/reproduction-plan.json \
  /path/to/ledger.json /path/to/sacct.psv --out /path/to/new/qualified.json
```

Review the reproduction elapsed times and completed iteration counts before planning matched jobs. Use the `budget` command with `--matched N --qualification /path/to/qualified.json` before each matched wave, with a fresh ledger and accounting snapshot. The [matched job template](../../configs/cfd-matched.sbatch) uses the same runner and a 35-minute limit. It requires `qualified.json` as its fourth positional argument.

Once the matched jobs reach terminal states, verify and freeze the retained CFD cases:

```bash
python -m airfoil_surrogate.cfd.campaign freeze /path/to/cohorts.json \
  /path/to/qualified.json /path/to/ledger.json /path/to/sacct.psv \
  --out /path/to/new/final-cohorts.json
```

The freeze verifies the expected ranked prefix in each group. A scientific rejection allows the next reserve, and a retry remains on the same geometry. It requires 30 qualified UIUC cases and 15 from each NACA family. The frozen cases contain the native mesh digests, completion records, and registered Slurm and `exp` IDs. An incomplete cohort cannot produce a final manifest for model inference.
