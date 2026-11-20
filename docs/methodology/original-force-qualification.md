# Qualify native-mesh inputs and force coefficients

Use the complete [original training stage](original-data-staging.md) and the pinned remeshed snapshot for the stored CL and CD. The audit uses the three cases from the prototype diagnostic, then the first nine other cases in the official training order. It checks the pinned label files and joins the selected labels against the training case IDs. The validation and test meshes are outside the audit.

Run the analytic constant-pressure, linear-pressure and wall-shear checks from the registered qualification revision. Run the 12-case command in the registered qualification output directory.

```bash
uv run --locked --extra forces pytest tests/test_original.py \
  -k 'original_force_adapter_on_known_fields or original_input_qualification'

python -m airfoil_surrogate.eval.qualify_forces \
  /path/to/train /path/to/airfrans_remeshed --out "$EXP_OUT/force-audit.json"
```

The reader checks finite fields and an exact wall-to-volume vertex match. The input audit checks positive fluid signed distance and zero normals off the wall. It locates the fluid cell adjacent to each wall segment, checks that the segment normal points into that cell, and checks that the point normals align with the incident segment normals. It uses the same numerical bounds as the CFD conversion check for the wall distance and speed (10⁻⁷) and normal lengths (10⁻⁶).

The force audit passes the unchanged native fields through the surrogate adapter and a direct `airfrans.Simulation` call. It records the signed discrepancies across total, pressure and viscous CL and CD. The proposed gates require every adapter contribution to agree with direct integration to absolute 10⁻⁸ and both direct totals to agree with the stored coefficients to absolute 10⁻⁵. The command writes the case-level numerical results and returns status 1 when either check fails.
