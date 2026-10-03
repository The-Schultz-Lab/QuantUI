# SLURM batch templates (QuantUI)

Reference scripts for operator-driven NCShare / Apptainer batch jobs.

## Files

| File | Purpose |
|------|---------|
| `quantui-batch.sbatch` | Reference `sbatch` template (partition must be set) |
| `quantui-gpu-test.sbatch` | GPU smoke test (operator use) |

QuantUI generates per-job scripts automatically when students submit from the UI.

## Submitting from a terminal (`quantui-batch`)

Students and research users submit over SSH from the login node. The login
node is for job submission, not computing, so `quantui-batch` never starts
the image there and never imports QuantUI: it is a standard-library Python
3.6+ script that writes the same job folder `SlurmBackend.prepare()` writes
(`request.json`, `submit.slurm`, and the job record the app reads), then
calls `sbatch`. The calculation runs inside the image on a compute node.
(`quantui-batch.sbatch` above is an unrelated hand-edited reference template.)

The values the launcher has to agree with QuantUI on (element table, method
list, limits, estimate factors, the batch-script template, the image's own
`python`) are copied in from QuantUI at install time. The small amount of logic
it reimplements (XYZ parsing, the charge/multiplicity check, the resource
estimate, job naming, the script) is held equal to QuantUI's by
`tests/test_batch_submit.py`. So **install it from the image it will submit
to, and reinstall whenever that image is replaced.** One shared install
serves every user; job folders are each user's own.

```bash
# operator, once per image, from an allocation (importing QuantUI on the login node is slow and can fail):
apptainer exec /opt/apps/containers/users/quantui.sif \
    quantui install-launcher /opt/apps/containers/users/bin --force
```

Users add that folder to `PATH` once, then on the login node:

```bash
quantui-batch check                                    # is everything this needs in place?
quantui-batch submit water.xyz --calc frequency --method B3LYP --basis def2-SVP --preopt
quantui-batch submit water.xyz --preset lab4-ir        # named settings (see Presets)
quantui-batch submit --from water-opt --calc frequency # start from that job's optimized geometry
quantui-batch estimate water.xyz --calc frequency      # cores / memory / time, submits nothing
quantui-batch status                                   # PENDING, RUNNING 40%, DONE, OUT_OF_MEMORY ...
quantui-batch results <job>                            # energy, imaginary modes, IR bands, excited states
quantui-batch log <job> -f                             # follow live.log of the latest attempt
quantui-batch rerun <job> --more-memory                # or --more-time, or any sbatch option
quantui-batch cancel <job>
```

- Inputs are `.xyz` files (`--calc` required unless a preset sets it;
  `--method`, `--basis`, `--charge`, `--mult`, `--solvent`, `--preopt`,
  `--option KEY=VALUE`) or request JSON. A charge/multiplicity that cannot
  fit the electron count, an unknown solvent, or a solvent on a calc type the
  batch worker runs gas-phase only (`tddft`, `frequency`, ...) is refused
  before anything is queued.
- **Presets** (`--preset NAME`; list with `quantui-batch presets`) are named
  settings: `calc`, `method`, `basis`, `charge`, `mult`, `solvent`, `preopt`,
  `options`, `description`. Read from `presets.json` beside the launcher
  (shared, operator-maintained; `install-launcher --force` leaves it alone),
  then `~/.quantui/batch-presets.json`, then `$QUANTUI_BATCH_PRESETS`; later
  files win, and command-line options override the preset.
- **Chained jobs**: `--from JOB` (a `geometry_opt` or `frequency` job, or one
  run with `--preopt`) takes JOB's molecule, charge, multiplicity, method and
  basis unless given, and the worker loads JOB's final geometry when the job
  starts (`quantui/backends/batch_chain.py`; never a PES-scan trajectory). If
  JOB is still queued or running, the new job gets
  `--dependency=afterok:<id> --kill-on-invalid-dep=yes`.
- **Duplicate guard**: a request identical to an earlier job (same
  calculation, settings and geometry) is refused unless that job failed;
  `--again` overrides.
- **Job limit**: at most `QUANTUI_MAX_CONCURRENT_JOBS` (default 2) QuantUI
  jobs per user may be queued or running, counted from `squeue` (jobs whose
  script is under the job root). `--queue-rest` submits the rest anyway, each
  with `--dependency=afterany:<oldest QuantUI job>`, so no more than the
  limit run at once.
- **Reruns** keep earlier attempts. `--more-memory` asks for twice the
  previous memory, or 1.5x the `sacct` MaxRSS if that is more; `--more-time`
  moves to the next `WALLTIME_OPTIONS` step (doubling past 48 h). Overrides
  carry over to later reruns. Submissions are logged in
  `<job>/.quantui-batch-jobs` (id, time, sbatch options).
- Job folders go to the user's job root: `QUANTUI_STAGING_DIR`, else
  **System Settings → SLURM job folder** (`compute.slurm_job_root` in
  `~/.quantui/settings.json`), else `~/.quantui/staging`.
- Every generated script carries `#SBATCH --comment=quantui`, so an operator
  can list everyone's QuantUI jobs: `squeue -h -o "%i|%u|%j|%T|%M|%k" | grep '|quantui$'`.
- `QUANTUI_BATCH_IMAGE` points the launcher at another image of the same
  kind (e.g. `/data/schultzlab/apptainers/quantui.sif`); for a different kind
  (the GPU image has a different `python`), install a launcher from it.
- The launcher drops the caller's `SLURM_*` before `sbatch`, so submitting
  from inside an allocation never leaks e.g. `SLURM_CPUS_PER_TASK` into the
  new job (the generated script sets `--ntasks`, not `--cpus-per-task`).
- Finished attempts appear in the app's **History** the next time the app
  starts (no Cluster Jobs tab or `QUANTUI_ENABLE_SLURM` needed).
- CPU image only for now: the generated script requests no GPU (`--gres`).

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
