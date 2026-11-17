# Original-mesh A100 timing pilot

Approved on 2026-11-17. Use one full A100 for at most one hour on the original AirfRANS full-task training split. This is a hardware measurement. The reported training campaign requires a separate decision.

Use seed 0, the repository's default Transolver, float32 and 32,000 nodes sampled without replacement per case and update. Time 100 updates after ten warm-up updates, cycling through three available training meshes in original manifest positions 1, 2 and 474. Fit the pilot normaliser on those three cases. The timing includes sampling, transfer, training and synchronisation. It excludes mesh preparation and disk I/O. Peak CUDA allocated and reserved memory cover the training loop including warm-up. The compressed one-cycle schedule exercises the optimiser. No pilot checkpoint is used for validation or geometry transfer.

The SLURM time limit is 15 minutes, with requeue disabled. The cluster runtime uses the available CUDA-enabled PyTorch environment. The output records its actual version. [Experiment record](../../experiments/2026-11-17-original-a100-pilot/experiment.json)
