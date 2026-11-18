# Stage the original AirfRANS meshes

The [dataset pin](../../manifests/airfrans_original.json) records the authors' processed `Dataset.zip`, its size and ETag, the hash of the ZIP member index, and the SHA-256 of the official split manifest. The member index records each path, uncompressed size and CRC-32. Staging checks the index and split before extracting any mesh. It verifies the CRC-32 while extracting and records a SHA-256 for each staged file. [Authors' dataset documentation](https://airfrans.readthedocs.io/en/latest/notes/dataset.html#downloading-the-dataset).

From the repository root, with a local copy of the pinned ZIP:

```bash
uv run --locked python -m airfoil_surrogate.data.stage_original /path/to/Dataset.zip \
  --partition train --out /path/to/new/train
uv run --locked python -m airfoil_surrogate.data.stage_original /path/to/Dataset.zip \
  --partition validation --out /path/to/new/validation
```

Staging reads only the mesh members in the requested partition. Use `--partition test` for the separate held-out evaluation after the checkpoint and baselines have been frozen. The 720 training cases, 80 validation cases and 200 test cases retain their official order. Use repeated `--case` arguments to stage a subset for qualification. Each stage requires a new directory and writes `stage.json` after all files have been extracted. The manifest records the partition, ordered case IDs, mesh sizes and digests.

Fit a normaliser with the complete training stage and a new output filename:

```bash
uv run --locked --extra forces python -m airfoil_surrogate.data.fit_original_normaliser \
  /path/to/train --out /path/to/normaliser.json
```

The fit checks the ordered 720 case IDs against the dataset pin before reading meshes and checks each mesh against its staged size and SHA-256. It fits the repository's pooled, float64 normaliser and compares its means and population standard deviations with independently merged centred moments (`rtol=1e-8`, `atol=1e-12`). The normaliser records the ordered fitting IDs and the SHA-256 of `stage.json`.
