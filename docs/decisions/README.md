# Decision records

Each **[agree]** item in [PLAN.md](../../PLAN.md) is settled here, with the user, before the first reported run that depends on it. Use one file per decision, named `YYYY-MM-DD-<slug>.md`. Records are append-only: to change a decision, write a new record that supersedes the old one and link the two.

## Index

| Item | PLAN.md section | Status | Record |
|---|---|---|---|
| Prototype data: AirfRANS_remeshed (reported-run data still open) | Data | accepted | [2026-11-17](2026-11-17-prototype-data-remeshed.md) |
| Leakage check and shape-held-out split | Data | evidence recorded (no shape repeats); awaiting agreement | [2026-11-17](2026-11-17-prototype-data-remeshed.md) |
| Data for reported runs (official processed vs remeshed) | Data | open | |
| Out-of-family set selection rule | Data | open | |
| Solver reproduction tolerance (CL, CD, surface pressure) | Solver reproduction | open | |
| Architectures and hyperparameters for reported runs | Models | open | |
| Metric set and which runs count | Evaluation harness | open | |
| Design-loop objectives (A lift-led, B CL/CD) | Design loop (Q3) | open | |
| Compute budget (GPU and OpenFOAM node-hours) | Compute budget | open | |

## Template

```markdown
# <Decision title>

- Date: YYYY-MM-DD
- Status: accepted | superseded by <link>
- PLAN.md item: <section>

## Decision
<one or two sentences>

## Options considered
- <option>: <main trade-off>

## Rationale
<why this option; assumptions it rests on>

## Evidence
<checks, exp run IDs, references>

## What would change it
<observation that would reopen this decision>
```
