# Decision records

Each **[agree]** item in [PLAN.md](../../PLAN.md) is settled here, with the user, before the first reported run that depends on it. Use one file per decision, named `YYYY-MM-DD-<slug>.md`. Records are append-only: to change a decision, write a new record that supersedes the old one and link the two.

## Index

| Item | PLAN.md section | Status | Record |
|---|---|---|---|
| Prototype data: AirfRANS_remeshed (reported-run data still open) | Data | accepted | [2026-11-17](2026-11-17-prototype-data-remeshed.md) |
| Leakage check and shape-held-out split | Data | accepted for this study | [2026-11-20](2026-11-20-reported-airfrans-data.md) |
| Data for reported runs (official processed vs remeshed) | Data | accepted for this study | [2026-11-20](2026-11-20-reported-airfrans-data.md) |
| Out-of-family set selection rule | Data | accepted for this study | [2026-11-20](2026-11-20-matched-airfoil-cohorts.md) |
| Solver reproduction tolerance (CL, CD, surface pressure) | Solver reproduction | accepted for this study | [2026-11-20](2026-11-20-openfoam-reproduction.md) |
| Architectures and hyperparameters for reported runs | Models | accepted for this study | [2026-11-20](2026-11-20-original-transolver-training.md) |
| Metric set and which runs count | Evaluation harness | accepted for this study | [2026-11-20](2026-11-20-original-transfer-measures.md) |
| Design-loop objectives (A lift-led, B CL/CD) | Design loop (Q3) | open | |
| Compute budget (GPU and OpenFOAM node-hours) | Compute budget | accepted for this study | [2026-11-20](2026-11-20-original-transfer-budget.md) |

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

The approved original-mesh hardware pilot is scoped separately from the reported-run decisions. [Pilot decision](2026-11-17-original-a100-pilot.md)

The [original-mesh Transolver and geometry-transfer protocol](../methodology/original-mesh-transfer-protocol.md) bundles the decisions for a one-seed, full-task study and was approved on 2026-11-20.
