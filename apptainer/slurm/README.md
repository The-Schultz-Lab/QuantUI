# SLURM batch templates (QuantUI)

Reference scripts for operator-driven NCShare / Apptainer batch jobs.

## Files

| File | Purpose |
|------|---------|
| `quantui-batch.sbatch` | Reference `sbatch` template (partition must be set) |
| `quantui-gpu-test.sbatch` | GPU smoke test (operator use) |

QuantUI generates per-job scripts automatically when students submit from the UI.

## Submitting from a terminal (`quantui-batch`)

On a cluster, QuantUI lives inside the Apptainer image and `sbatch` lives on
the host. The `quantui-batch` launcher bridges the two: it runs
`quantui submit --prepare-only` inside the image to write the job folder
(same layout, same resource estimate as the app), then submits
`submit.slurm` with the host's `sbatch`. (`quantui-batch.sbatch` above is an
unrelated hand-edited reference template.)

Install it once per user, from the image you want jobs to run in:

```bash
apptainer exec /path/to/quantui.sif quantui install-launcher   # writes ~/bin/quantui-batch
```

The launcher is standard-library Python 3.6+ and remembers that image and the
job root. Reinstall (`--force`) after the image moves or the job root changes;
`QUANTUI_BATCH_IMAGE` / `QUANTUI_STAGING_DIR` override both at run time.

```bash
quantui-batch submit water.xyz --calc frequency --method B3LYP --basis def2-SVP --preopt
quantui-batch estimate water.xyz --calc frequency      # cores / memory / time, submits nothing
quantui-batch status                                   # recent jobs: PENDING, RUNNING 40%, DONE, OUT_OF_MEMORY ...
quantui-batch log <job> -f                             # follow live.log of the latest attempt
quantui-batch rerun <job> --mem=64G --time=24:00:00    # new attempt; sbatch options override the script
quantui-batch cancel <job>
```

- Inputs are `.xyz` files (`--calc` required; `--method`, `--basis`,
  `--charge`, `--mult`, `--solvent`, `--preopt`, `--option KEY=VALUE`) or
  request JSON. A charge/multiplicity that cannot fit the electron count is
  refused before anything is queued.
- At most `QUANTUI_MAX_CONCURRENT_JOBS` (default 2) QuantUI jobs per user may
  be queued or running, counted from `squeue` (jobs whose script is under the
  job root). Raise it for heavy users.
- The prepare step runs with one BLAS/OpenMP thread (a login node's per-user
  thread limit has broken numpy imports inside the image before). If
  `apptainer` is not on `PATH` (or with `QUANTUI_BATCH_PREPARE=srun`), it runs
  through a 1-CPU, 2 GB, 5-minute `srun` on `QUANTUI_SLURM_PARTITION`
  (default `common`) instead.
- Submitting from inside a job (an OnDemand Shell session is one) is safe:
  the launcher drops the caller's `SLURM_*` variables before `sbatch`/`srun`,
  so e.g. the shell's `SLURM_CPUS_PER_TASK` never reaches the new job.
- Finished attempts appear in the app's **History** the next time the app
  starts (no Cluster Jobs tab or `QUANTUI_ENABLE_SLURM` needed).
- CPU image only for now: the generated script requests no GPU (`--gres`), so
  a GPU image run this way computes on CPU.

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
