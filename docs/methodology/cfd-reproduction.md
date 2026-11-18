# Qualify the original NACA CFD setup

Use the pinned official split manifest and the proposed operating-point rule to record the three training cases for CFD reproduction:

```bash
uv run --locked --extra forces python -m airfoil_surrogate.cfd.reproduction plan \
  /path/to/Dataset/manifest.json --out /path/to/new/reproduction-plan.json
```

This reads the official case IDs and checks their order. It records the original rounded inlet velocity and angle of attack, NACA parameters, iteration limit and pinned generator revision. The plan uses no reference flow fields or model predictions.

After the approved OpenFOAM jobs have completed and exported their native mesh files in the original processed layout, verify each case against its pinned training reference:

```bash
python -m airfoil_surrogate.cfd.reproduction verify \
  /path/to/reproduction-plan.json /path/to/train CASE_ID \
  /path/to/native/internal.vtu /path/to/native/aerofoil.vtp \
  /path/to/log.checkMesh /path/to/log.simpleFoam \
  /path/to/postProcessing/forceCoeffs1/0/coefficient.dat \
  --out "$EXP_OUT/reproduction-check"
```

Use the `forces` and Matplotlib dependencies in the verification environment. The verifier requires the monitor's last 1,000 iterations to be contiguous and end at the final iteration in the completed `simpleFoam` log. It reads `Cl` and `Cd` by column name. Its coefficient comparisons use the AirfRANS integration routine on both native meshes. Its pressure comparison follows the wall connectivity and interpolates the upper and lower surfaces independently at 99 fixed chord stations. The verifier writes the overlaid plot, the values used in it, the comparisons and source digests. All three cases must pass the proposed gates before matched CFD. [Pinned generator and monitoring conventions](https://github.com/Extrality/NACA_simulation/tree/2dbf00c5ecbb8d6cef720db1e72c1fe40307eeab).
