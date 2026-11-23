# Reference mesh requalification

- Date: 2026-11-23
- Status: approved by the user
- PLAN.md items: Solver reproduction and Compute budget

Run the same three original NACA cases on Eureka2 parallel scratch with the pinned OpenFOAM v2112 generator and settings. Each solve has a 60-minute wall limit on 16 cores. The approved requalification cap is 50 allocated CPU-core-hours and zero GPU-hours, including verification and recovery. This changes the reproduction wall limit in the [original decision](2026-11-20-openfoam-reproduction.md).

For the first two reference cases, allow the single aspect-ratio check failure only when the regenerated mesh, the flagged cell set and the reported count and maximum match the audited mesh in the [pinned manifest](../../manifests/airfrans_reproduction_mesh.json). The released meshes contain all the flagged cells with the same connectivity. The runner and output verifier both check the digests of the uncompressed mesh files. The third reference case and the matched cohorts use the standard mesh check. The lift, drag, surface pressure and force-convergence criteria in the [original protocol](../methodology/original-mesh-transfer-protocol.md) remain in force.

The first two mesh failures are recorded in [run `20261120T232532Z-c620`](../../experiments/2026-11-20-original-mesh-transfer/runs/20261120T232532Z-c620/run.json) and [run `20261120T233024Z-9c8e`](../../experiments/2026-11-20-original-mesh-transfer/runs/20261120T233024Z-9c8e/run.json). The third reached its allocation limit near the end of the solver schedule and retained no final fields. [Run `20261120T233026Z-51a2`](../../experiments/2026-11-20-original-mesh-transfer/runs/20261120T233026Z-51a2/run.json)
