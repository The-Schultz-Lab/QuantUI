# Security Policy

QuantUI is an educational teaching tool designed for classroom and local
research use. By default it runs calculations inside your own Python session —
there is no hosted server and no user accounts, and results, logs, settings
and checkpoints stay on the machine (or cluster account) you run it on. The optional SLURM batch mode
writes job folders under your own cluster home and submits them with your own
`sbatch`.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.11.x  | Yes       |
| < 0.11  | No        |

## Reporting a Vulnerability

If you find a security issue (e.g. a path-traversal bug, an unsafe use of
`subprocess`, or a dependency with a known CVE), please **do not open a public
GitHub issue**.

Instead, report it privately via one of these channels:

- **GitHub private vulnerability reporting** — use the
  [Security tab](https://github.com/The-Schultz-Lab/QuantUI/security/advisories/new)
  on this repository (preferred).
- **Email** — contact the lab maintainer through the GitHub profile
  [@The-Schultz-Lab](https://github.com/The-Schultz-Lab).

Please include:

1. A short description of the issue and its potential impact
2. Steps to reproduce (or a minimal proof-of-concept)
3. The version of QuantUI affected
4. Your suggested fix, if you have one

We aim to acknowledge reports within **5 business days** and to release a fix
within **30 days** for confirmed vulnerabilities.

## Scope

This project is a local teaching tool. The threat model is limited — there is
no multi-user server and no user accounts. It does keep **persistent data in
the user's own account**, protected only by its file permissions: saved
results (`./results/` by default, `QUANTUI_RESULTS_DIR` in the Apptainer
image and Viewer), and under `~/.quantui/` logs (`logs/perf_log.jsonl`,
`logs/event_log.jsonl`), user settings (`settings.json`), calculation
checkpoints (`checkpoints/`), SLURM job folders and staging (`jobs/`,
`staging/`) and the generated launcher notebook. The log, settings and
checkpoint paths can be redirected with `QUANTUI_LOG_DIR`,
`QUANTUI_SETTINGS_PATH` and `QUANTUI_CHECKPOINT_DIR`.

The primary security concerns are:

- **Access to the running app.** `quantui run app`, `quantui view`, the native
  launchers and the Apptainer image's `app` runscript bind Voilà to
  `127.0.0.1` and require a per-run token (`QUANTUI_APP_TOKEN` when set,
  otherwise a fresh random one); the printed/opened URL carries it
  (`?token=...`). Standalone Voilà reads no token from the environment, so on
  a shared host the token is visible to other local users in the process
  list (`ps`). `launch-native-jupyter.sh` uses `JUPYTER_TOKEN` (off the
  command line). The Apptainer image's default JupyterLab mode keeps
  Jupyter's own token handling. Anyone holding the token can drive the
  session as you — read History, submit SLURM jobs with your account.
- **Untrusted files the app reads:** the Viewer's `.zip` upload of a results
  folder (`quantui/app_viewer.py`), structure-file upload
  (`quantui/structure_upload.py`: XYZ/PDB/CIF/MOL/SDF/MOL2/Gaussian), and
  ingestion of SLURM job folders and result data written outside the session
  (`quantui/backends/slurm_ingest.py`, `quantui/results_storage.py`).
- **Path traversal** in file I/O (the uploads/ingest above, molecule XYZ
  parsing, script and bundle export)
- **Unsafe subprocess calls** — PySCF runs in-process via the Python API, but
  `subprocess` is used for SLURM (`sbatch`, `squeue`, `sacct`, `scancel`; see
  `quantui/backends/slurm.py` and the `quantui-batch` launcher), for starting
  Voilà (`quantui run app` / `quantui view`), and for small CLI and diagnostic
  helpers. Job-folder paths and resource requests for SLURM are validated in
  `quantui/backends/cluster_security.py`.
- **Dependency vulnerabilities** in the scientific Python stack (numpy, pyscf,
  ase, etc.) — monitored via Dependabot

Issues that are explicitly **out of scope**:

- Vulnerabilities that require physical access to the machine running the
  notebook
- Vulnerabilities in Jupyter/JupyterHub itself (report those upstream)
- Social-engineering or phishing attacks

## Dependencies

QuantUI relies on PySCF, ASE, NumPy, Matplotlib, ipywidgets, and
py3Dmol. Security advisories for these packages are tracked automatically
via Dependabot. If you become aware of a critical CVE in one of these
dependencies before Dependabot picks it up, please report it using the
channels above.
