# QuantUI CLI

QuantUI ships a small command-line toolkit for inspecting state and
generating reports from outside the notebook. After installing the
package (`pip install -e .` or `pip install quantui`), the `quantui`
command is on your `PATH`.

```bash
quantui --help
```

The CLI is meant to *complement* the Voilà app — not replace it. Reach
for the CLI when you want to:

- check what the app has been doing without opening a notebook
- confirm GPU offload is wired correctly before starting a long run
- generate a usage / GPU-speedup report you can share or pin to a tab
- script log inspection or analytics into a shell pipeline / cron job

The CLI never touches your live calculations or notebook server. All
commands are read-only against `~/.quantui/` (or whatever
`QUANTUI_LOG_DIR` points at).

---

## Command reference

| Command | What it does |
| --- | --- |
| [`quantui log tail`](#quantui-log-tail) | Print recent events from `event_log.jsonl` |
| [`quantui gpu check`](#quantui-gpu-check) | Probe GPU-offload availability and explain failures |
| [`quantui analytics build`](#quantui-analytics-build) | Build an HTML usage dashboard from `perf_log.jsonl` |

---

## `quantui log tail`

Print the last *N* entries from the QuantUI event log
(`~/.quantui/logs/event_log.jsonl`). Each event is rendered on one
line as `timestamp  event_type  message  k=v k=v ...`, so the output
is grep-friendly.

### Flags

| Flag | Default | Description |
| --- | --- | --- |
| `-n N` | `20` | Number of most-recent events to print |

### Examples

```bash
# Last 20 events
quantui log tail

# Last 50 events
quantui log tail -n 50

# Find every GPU-related event
quantui log tail -n 500 | grep -i gpu

# Watch the most recent error
quantui log tail -n 200 | grep -i error | tail -5
```

### Sample output

```
2026-05-25T13:55:22.421910+00:00  viz_route_decision  task=molecule_preview pref=auto chosen=py3dmol reason=auto -> task primary (py3dmol)
2026-05-25T13:55:22.470028+00:00  startup             QuantUI 0.3.0 started (viz backend pref=auto)
2026-05-25T14:08:14.102544+00:00  calc_done           B3LYP/STO-3G on H2O  elapsed_s=1.2 converged=True gpu_used=True gpu_name=NVIDIA GeForce RTX 4050 Laptop GPU
```

### Notes

- The event log auto-prunes entries older than 7 days on every write,
  so `tail` always reflects the active week.
- Output goes to stdout; "log is empty" / "log not found" notices go to
  stderr so they don't pollute pipelines.
- Exit code: always `0` (even when no events exist — the absence of
  events is not an error).

---

## `quantui gpu check`

Probe whether QuantUI's GPU offload path is functional in the current
environment. This is the canonical one-liner for verifying that
`gpu4pyscf` + `cupy` are installed correctly and that
`is_gpu_available()` will return `True` when the app runs.

### Flags

None.

### Examples

```bash
# Is GPU offload working right now?
quantui gpu check

# Use in a shell condition
if quantui gpu check; then
    echo "GPU mode"
else
    echo "Falling back to CPU"
fi

# Diagnose a CI machine
QUANTUI_DISABLE_GPU=1 quantui gpu check   # confirms env-var path
```

### Sample output

**When GPU is available:**

```
GPU offload available: NVIDIA GeForce RTX 4050 Laptop GPU
```

(exit code `0`)

**When GPU is unavailable**, the command prints a reason so you know
where to look next:

```
GPU offload not available
  reason: gpu4pyscf not installed (see README → 'Optional: GPU acceleration')
```

```
GPU offload not available
  reason: QUANTUI_DISABLE_GPU is set in the environment
```

```
GPU offload not available
  reason: cupy reports 0 CUDA devices
```

```
GPU offload not available
  reason: cupy/gpu4pyscf import succeeded but probe raised — run
  `python -c "import cupy; cupy.show_config()"` to inspect
```

(all return exit code `1`)

### Notes

- Detection is cached for the lifetime of QuantUI's runtime (so the
  Voilà app doesn't re-probe on every result-card render); the CLI
  clears that cache before probing so each invocation reflects the
  current state — useful right after a `pip install`.
- Returns exit `1` rather than raising, so the command is safe to use
  in `if ...; then ... fi` and `&& ...` chains.

---

## `quantui analytics build`

Build a self-contained HTML analytics dashboard from
`~/.quantui/logs/perf_log.jsonl` and write it to a file you can open
in any browser.

The dashboard contains:

- **Overview cards** — total runs, total compute time, GPU vs CPU run
  counts, unique molecules / methods / basis sets used.
- **GPU vs CPU speedup table** — for every `(method, basis, formula)`
  tuple that has runs on *both* devices, the median CPU time, median
  GPU time, and the speedup factor. Sorted best-speedup first.
- **Method usage** — bar chart of run counts per method.
- **Calc-type distribution** — bar chart of run counts per calculation
  type.
- **Recent timeline** — scatter of `elapsed_s` over time, coloured by
  compute device (CPU grey, GPU green, Unknown light grey for
  pre-2026-05-25 records that don't yet have device info).

Plotly's JavaScript is inlined into the HTML, so the file works
offline and can be emailed, attached to a writeup, or pinned to a
browser tab.

### Flags

| Flag | Default | Description |
| --- | --- | --- |
| `-o PATH`, `--output PATH` | `~/.quantui/dashboard.html` | Output HTML path |
| `--open` | off | After writing, open the dashboard in the default browser (WSL-aware — uses `wslview` / `explorer.exe` on WSL) |

### Examples

```bash
# Build the dashboard at the default location
quantui analytics build

# Build and immediately open it in the browser
quantui analytics build --open

# Write somewhere shareable
quantui analytics build -o ~/Desktop/quantui-report.html

# Build into a shared folder + open
quantui analytics build -o ~/projects/lab-share/quantui-report.html --open
```

### Sample output

```
Wrote /home/youruser/.quantui/dashboard.html
```

With `--open`, the CLI picks the right opener for your environment:

- **WSL**: tries `wslview` first (bundled with the `wslu` package),
  then falls back to `explorer.exe`. Both delegate to your **Windows
  default browser** via WSL interop — no Linux-side browser install
  needed. If neither is available, `sudo apt install wslu` fixes it
  in one step.
- **Linux native**: stdlib `webbrowser.open` (which uses `xdg-open`).
- **macOS / Windows native**: stdlib `webbrowser.open`.

If no opener succeeds — e.g. a headless container with no display —
you'll see:

```
Wrote /home/youruser/.quantui/dashboard.html
(could not auto-open browser — open /home/youruser/.quantui/dashboard.html manually)
```

The exit code stays `0` either way — the dashboard was written
successfully; only the auto-open is best-effort.

### Notes

- **Empty perf log**: if `perf_log.jsonl` doesn't exist yet, the
  command prints `(perf log is empty — run a calculation first)` to
  stderr and exits `0`. No file is written.
- **Old records with no GPU info**: records written before session 55
  (2026-05-25) don't have `gpu_used`. The dashboard counts those in a
  separate "Unknown" device bucket rather than assuming CPU — that
  keeps the GPU-vs-CPU speedup table honest.
- **Speedup table empty?** It only shows tuples that have runs on
  *both* devices. After enabling GPU, re-run any prior CPU calc on
  the GPU to populate at least one row.

---

## Environment variables

| Variable | Effect |
| --- | --- |
| `QUANTUI_LOG_DIR` | Override the default `~/.quantui/logs/` location. The dashboard's default output (`~/.quantui/dashboard.html`) follows: it lives one level up from the active `QUANTUI_LOG_DIR`. |
| `QUANTUI_DISABLE_GPU` | Force CPU mode even when gpu4pyscf is installed. `quantui gpu check` reports this as the reason. Accepted truthy values: `1`, `true`, `True`. |

---

## Common workflows

### Verify GPU is wired before a long run

```bash
quantui gpu check && voila notebooks/molecule_computations.ipynb
```

If `gpu check` exits non-zero, the Voilà launch is skipped and the
reason was printed to stderr.

### Quick "what happened in my last session?"

```bash
quantui log tail -n 100 | grep -E "calc_done|calc_error|startup"
```

### After a benchmarking run, open the report

```bash
quantui analytics build --open
```

The dashboard opens; the speedup table summarises everything across
runs without you needing to remember which calc ran where.

### Plumbing into cron / CI

```bash
# Daily snapshot, no auto-open (headless)
quantui analytics build -o /var/reports/quantui-$(date +%F).html
```

---

## Adding a new subcommand

Each verb is one `_cmd_<verb>(args: argparse.Namespace) -> int` in
[`quantui/cli.py`](https://github.com/The-Schultz-Lab/QuantUI/blob/main/quantui/cli.py) plus a registration in
`_build_parser`. The pattern is short by design — `gpu check`,
`log tail`, and `analytics build` all fit in well under 50 lines of
production code apiece. See the module docstring for the contract.

Tests live in [`tests/test_cli.py`](https://github.com/The-Schultz-Lab/QuantUI/blob/main/tests/test_cli.py) — every
subcommand should cover its happy path, its empty/missing-data path,
and any flag-specific behavior (e.g. `--open` was tested against both
a successful `webbrowser.open` and a failed one).
