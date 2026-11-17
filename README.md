# AirfRANS geometry-aware neural surrogate

Mesh-native neural surrogates for 2D RANS airfoil flow, trained and benchmarked on AirfRANS. Beyond the standard benchmark, the project tests two things that matter in industrial use: accuracy on airfoil shapes outside the training family, and whether design improvements found with the surrogate survive a real OpenFOAM check.

**Status:** M0 (setup) in progress. There are no results yet. The research plan, questions and milestones are in [PLAN.md](PLAN.md).

## Data

This project uses the AirfRANS dataset (Bonnet et al. 2022), which is licensed under the [Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/). The data is not redistributed here.

```bibtex
@inproceedings{bonnet2022airfrans,
  title     = {Airf{RANS}: High Fidelity Computational Fluid Dynamics Dataset for Approximating {Reynolds}-Averaged {Navier}--{Stokes} Solutions},
  author    = {Florent Bonnet and Jocelyn Ahmed Mazari and Paola Cinnella and Patrick Gallinari},
  booktitle = {Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year      = {2022},
  url       = {https://arxiv.org/abs/2212.07564}
}
```

## Development

```bash
uv sync
uv run pre-commit install
uv run ruff check . && uv run ruff format --check .
```

## Licence

Code: MIT (see [LICENSE](LICENSE)). Data: ODbL 1.0, from the upstream AirfRANS dataset.
