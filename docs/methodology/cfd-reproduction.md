# Qualify the original NACA CFD setup

Use the pinned official split manifest and the proposed operating-point rule to record the three training cases for CFD reproduction:

```bash
uv run --locked --extra forces python -m airfoil_surrogate.cfd.reproduction plan \
  /path/to/Dataset/manifest.json --out /path/to/new/reproduction-plan.json
```

This reads the official case IDs and checks their order. It records the original rounded inlet velocity and angle of attack, NACA parameters, iteration limit and pinned generator revision. The plan uses no reference flow fields or model predictions.

After the approved solver job has completed, select the volume and aerofoil patch from the same final `foamToVTK` export. Use the common conversion for reproduction, control and UIUC cases:

```bash
python -m airfoil_surrogate.cfd.convert \
  /path/to/VTK/final/internal.vtu /path/to/VTK/final/boundary/aerofoil.vtp \
  --out "$EXP_OUT/native"
```

The converter retains whole cells intersecting the AirfRANS clip box, slices the volume and wall at z = 0.5 m, and computes signed distances against the three-dimensional wall surface. It requires exact vertex matching between the sliced meshes, checks distance sign, planar unit normals and no-slip, and records the point-field differences between the wall patch and the volume at those vertices. It writes a native `internal.vtu` and `aerofoil.vtp` with a conversion manifest containing input and output hashes. [AirfRANS preprocessing description](https://airfrans.readthedocs.io/en/latest/notes/dataset.html#raw-versus-pre-processed-dataset).

Verify each reproduction against its pinned training reference:

```bash
python -m airfoil_surrogate.cfd.reproduction verify \
  /path/to/reproduction-plan.json /path/to/train CASE_ID \
  /path/to/native/internal.vtu /path/to/native/aerofoil.vtp \
  /path/to/log.checkMesh /path/to/log.simpleFoam \
  /path/to/postProcessing/forceCoeffs1/0/coefficient.dat \
  --out "$EXP_OUT/reproduction-check"
```

Use the `forces` and Matplotlib dependencies in the verification environment. The verifier requires the monitor's last 1,000 iterations to be contiguous and end at the final iteration in the completed `simpleFoam` log. It reads `Cl` and `Cd` by column name. Its coefficient comparisons use the AirfRANS integration routine on both native meshes. Its pressure comparison follows the wall connectivity and interpolates the upper and lower surfaces independently at 99 fixed chord stations. The verifier writes the overlaid plot, the values used in it, the comparisons and source digests. All three cases must pass the proposed gates before matched CFD. [Pinned generator and monitoring conventions](https://github.com/Extrality/NACA_simulation/tree/2dbf00c5ecbb8d6cef720db1e72c1fe40307eeab).
