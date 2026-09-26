# SLURM batch templates (QuantUI)

Reference scripts for operator-driven NCShare / Apptainer batch jobs.

## Files

| File | Purpose |
|------|---------|
| `quantui-batch.sbatch` | Reference `sbatch` template (partition must be set) |
| `quantui-gpu-test.sbatch` | GPU smoke test (operator use) |

QuantUI generates per-job scripts automatically when students submit from the UI.

## Job folders

Each submission gets its own folder under the job root (default
`~/.quantui/staging`; set it in **System Settings → SLURM job folder** or with
`QUANTUI_STAGING_DIR`):

```
<job root>/H2O_opt_B3LYP_def2-SVP/        # job name = SLURM --job-name
    request.json   submit.slurm
    slurm-812345.out  slurm-812345.err   # SLURM's own output, one pair per job
    attempt-01_job812345/                # first run
        live.log  progress.json  ...
    attempt-02_job812399/                # a rerun, never overwrites attempt-01
        result.json  orbitals.npz  H2O_opt_B3LYP_def2-SVP.molden  ...
    latest -> attempt-02_job812399
```

- The job name defaults to `<formula>_<calc>_<method>_<basis>`; the optional
  **Job name** field on the Calculate tab (or `quantui submit --job-name`)
  overrides it. A name already in use gets `_2`, `_3`, ....
- `submit.slurm` creates a new `attempt-NN_job<SLURM id>/` folder every time it
  runs, so both **Resubmit** on the Cluster Jobs tab and a hand-run
  `sbatch submit.slurm` from the job folder start a fresh attempt. Earlier
  attempts are never overwritten, and checkpoints let a rerun resume.
- Every successful attempt appears in **History**, marked `🖥 SLURM <job id>·a<attempt>`,
  including hand-run ones (picked up on the next Cluster Jobs refresh).
  Failed attempts stay in the job folder only.
- Jobs submitted before per-job folders existed keep their
  `~/.quantui/staging/<request_id>/` layout and cannot be resubmitted in place.

Status polling uses **`squeue`** for active jobs and **`sacct`** for terminal state, exit code, and cancel confirmation. Cluster Jobs **Remove** clears terminal registry rows without deleting staging logs.

## Operator environment variables

See the [NCShare SLURM batch runbook](https://github.com/The-Schultz-Lab/QuantUI-development-tracking/blob/main/TODO/runbooks/NCShare-SLURM-batch-runbook.md) for the full operator matrix. Key overrides:

| Variable | Default | Purpose |
|----------|---------|---------|
| `QUANTUI_ENABLE_SLURM` | *(unset — off)* | Show **SLURM batch (cluster)** in Settings and allow cluster dispatch. Requires `sbatch` on PATH. Leave unset in student CPU images; set on instructor/test profiles when validating NCShare. |
| `QUANTUI_MAX_CONCURRENT_JOBS` | `2` | Active SLURM job cap |
| `QUANTUI_SLURM_SUBMIT_COOLDOWN_S` | `30` | Min seconds between submits (`0` disables) |
| `QUANTUI_SLURM_STALE_NO_ID_S` | `600` | Stale registry rows without SLURM id |
| `QUANTUI_SLURM_CANCEL_CONFIRM_S` | `30` | Seconds to wait for `scancel` confirmation via `sacct` |
| `QUANTUI_SLURM_PARTITION` | `common` | Default `#SBATCH` partition |
| `QUANTUI_BATCH_IMAGE` | `~/quantui-gpu.sif` | Apptainer image for batch worker |
| `QUANTUI_STAGING_DIR` | *(unset — System Settings value, else `~/.quantui/staging`)* | Job folder root. Overrides the System Settings value and locks that field. A root outside `$HOME` is bound into Apptainer automatically. |

## Operator runbook

See the planning repo runbook:

[NCShare SLURM batch runbook](https://github.com/The-Schultz-Lab/QuantUI-development-tracking/blob/main/TODO/runbooks/NCShare-SLURM-batch-runbook.md)

## Worker entrypoint

```bash
python -m quantui.backends.worker --request /path/to/job/request.json \
    --attempt-dir /path/to/job/attempt-01_job812345
```

`submit.slurm` passes `--attempt-dir` itself. Without it, outputs go next to
`request.json` (the pre-job-folder layout).

Supported calc types: `single_point`, `geometry_opt`, `frequency`, `tddft`, `nmr`, `pes_scan`, `reorganization_energy`.
