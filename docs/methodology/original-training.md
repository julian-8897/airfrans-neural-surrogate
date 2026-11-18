# Train Transolver on original AirfRANS

After protocol approval and data and model qualification, use the [staged full-task partitions and normaliser](original-data-staging.md) in a registered job on one full A100.

```bash
python -m airfoil_surrogate.train_original \
  /path/to/train /path/to/validation /path/to/normaliser.json \
  --out "$EXP_OUT/training"
```

The command accepts the pinned training and validation stages. It checks the normaliser's training provenance and hashes each file before loading it. Original-mesh features are packed once in CPU memory. Each update draws a wall-retaining sample and transfers it to the GPU. Validation uses fixed samples from an independent seeded generator. The checkpoint is selected by the lowest validation score, with the earlier epoch breaking an exact tie.

After a scheduler interruption, start a new tracked attempt with the same code, environment and pinned inputs. Pass the prior attempt's `resume.pt` and a new output directory:

```bash
python -m airfoil_surrogate.train_original \
  /path/to/train /path/to/validation /path/to/normaliser.json \
  --resume /path/to/prior/training/resume.pt --out "$EXP_OUT/training"
```

`resume.pt` is replaced atomically only after a complete epoch. It records model, optimiser, scheduler and RNG states, all completed metrics and the best validation weights so far. At completion, `model.pt` contains the selected weights, `selection.json` records eligible validation scores and the selected checkpoint hash, and `metrics.jsonl`, `run.json` and `normaliser.json` preserve metrics and provenance.

## Fit the baselines and freeze the evaluation inputs

Fit the field and coefficient baselines on the complete training stage in a registered run with the `forces` dependencies installed:

```bash
python -m airfoil_surrogate.eval.original_baselines \
  /path/to/train --out "$EXP_OUT/baselines.json"
```

The field baseline averages each region within each training case, then weights the cases equally. The coefficient baseline fits one quadratic ridge per NACA family using the original-mesh reference CL and CD. Its feature indices, training means and scales, and fitted weights are stored with the ordered fitting cases.

Once training and the baselines are complete, freeze the evaluation inputs before staging the official test meshes:

```bash
python -m airfoil_surrogate.eval.freeze_original \
  /path/to/completed/training /path/to/baselines.json --out "$EXP_OUT/frozen.json"
```

The freeze command checks that training and the baselines used the same pinned training cases. It verifies the complete validation schedule and checkpoint selection, then binds the baseline, checkpoint, normaliser, run metadata and recorded metrics by SHA-256.
