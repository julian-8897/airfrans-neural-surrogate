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

Keep CPU preparation and verification jobs in a separate snapshot with the same fields. Pass it to `budget`, `qualify` and `freeze` with `--preparation-accounting` so these allocations count towards the same cap. Use the full reserved wall time when proposing a new preparation allocation.

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

On Eureka2, the `cfd-campaign` mode of [the preparation template](../../configs/original-eureka2-preparation.sbatch) takes the scratch Python executable followed by the campaign command and its inputs. It writes `campaign.json` to that allocation's scratch output directory. Use `--reference-requalification` when the ledger includes the approved 60-minute reference jobs. Pass `--cpu-core-hours 675` for the [revised matched campaign cap](../decisions/2026-11-23-matched-cpu-cap.md).

Once the matched jobs reach terminal states, verify and freeze the retained CFD cases:

```bash
python -m airfoil_surrogate.cfd.campaign freeze /path/to/cohorts.json \
  /path/to/qualified.json /path/to/ledger.json /path/to/sacct.psv \
  --out /path/to/new/final-cohorts.json
```

The freeze verifies the expected ranked prefix in each group. A scientific rejection allows the next reserve, and a retry remains on the same geometry. It requires 30 qualified UIUC cases and 15 from each NACA family. The frozen cases contain the native mesh digests, completion records, and registered Slurm and `exp` IDs. An incomplete cohort cannot produce a final manifest for model inference.
