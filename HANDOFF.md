# Last session (2026-11-17)

Implemented and exercised the AirfRANS_remeshed validation harness against the saved Transolver smoke checkpoint `runs/20261117T201901Z-smoke-full-s0`. The evaluator reads the saved validation list and normaliser and writes per-case and summary metrics to a new directory. Its 80-case CPU check reproduced the final logged validation MSE to within 1 × 10⁻⁷ per channel. No training was launched.

- `uv run --locked --extra forces python -m airfoil_surrogate.eval runs/20261117T201901Z-smoke-full-s0 --forces --out runs/20261117-prototype-evaluation-smoke-s0` reproduces the validation check.
- `uv run --locked --extra forces python scripts/check_force_labels.py /path/to/Dataset 0 194 437` compares stored labels with integration on both meshes. The audit JSON is in `runs/20261117-prototype-force-labels/check.json`. It records the original patch checksums. The available original probe supplied case 0, and cases 194 and 437 came from the pinned mirror.
- `uv run --locked --extra forces pytest -q` includes the analytic force and field checks, the data tests and the CPU train-to-eval test. The [prototype methodology](docs/methodology/prototype-evaluation.md) gives the metric definitions and ground-truth findings.

No processes are in flight. The reported data, metric protocol and compute budget still need agreement before experiment registration. The three-case force check supports original-mesh provenance for those labels and shows a drag deficit when integrating the remeshed ground truth. Transolver parity with the official implementation is still open.
