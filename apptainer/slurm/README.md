# SLURM batch templates (QuantUI)

Reference scripts for operator-driven NCShare / Apptainer batch jobs.

## Files

| File | Purpose |
|------|---------|
| `quantui-batch.sbatch` | Reference `sbatch` template (partition must be set) |
| `quantui-gpu-test.sbatch` | GPU smoke test (operator use) |

QuantUI generates per-job scripts automatically under `~/.quantui/staging/<request_id>/submit.slurm` when students submit from the UI.

## Operator environment variables

See the [NCShare SLURM batch runbook](https://github.com/The-Schultz-Lab/QuantUI-development-tracking/blob/main/TODO/runbooks/NCShare-SLURM-batch-runbook.md) for the full operator matrix. Key overrides:

| Variable | Default | Purpose |
|----------|---------|---------|
| `QUANTUI_MAX_CONCURRENT_JOBS` | `2` | Active SLURM job cap |
| `QUANTUI_SLURM_SUBMIT_COOLDOWN_S` | `30` | Min seconds between submits (`0` disables) |
| `QUANTUI_SLURM_STALE_NO_ID_S` | `600` | Stale registry rows without SLURM id |
| `QUANTUI_SLURM_CANCEL_CONFIRM_S` | `30` | Seconds to wait for `scancel` confirmation via `sacct` |
| `QUANTUI_SLURM_PARTITION` | `common` | Default `#SBATCH` partition |
| `QUANTUI_BATCH_IMAGE` | `~/quantui-gpu.sif` | Apptainer image for batch worker |

## Operator runbook

See the planning repo runbook:

[NCShare SLURM batch runbook](https://github.com/The-Schultz-Lab/QuantUI-development-tracking/blob/main/TODO/runbooks/NCShare-SLURM-batch-runbook.md)

## Worker entrypoint

```bash
python -m quantui.backends.worker --request /path/to/staging/request.json
```

Supported calc types: `single_point`, `geometry_opt`, `frequency`, `tddft`, `nmr`, `pes_scan`, `reorganization_energy`.
