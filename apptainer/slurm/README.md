# SLURM batch templates (QuantUI)

Reference scripts for operator-driven NCShare / Apptainer batch jobs.

## Files

| File | Purpose |
|------|---------|
| `quantui-batch.sbatch` | Reference `sbatch` template (partition must be set) |
| `quantui-gpu-test.sbatch` | GPU smoke test (operator use) |

QuantUI generates per-job scripts automatically under `~/.quantui/staging/<request_id>/submit.slurm` when students submit from the UI.

## Operator runbook

See the planning repo runbook:

[NCShare SLURM batch runbook](https://github.com/The-Schultz-Lab/QuantUI-development-tracking/blob/main/TODO/runbooks/NCShare-SLURM-batch-runbook.md)

## Worker entrypoint

```bash
python -m quantui.backends.worker --request /path/to/staging/request.json
```

Supported calc types: `single_point`, `geometry_opt`, `frequency`.
