# CLAUDE.md — QuantUI

Onboarding for Claude Code (and other agents) working in this repo, **local or
cloud**. This is the short "where am I, what can I do, where do the plans live"
layer. Deep architecture lives in
[`.github/copilot-instructions.md`](.github/copilot-instructions.md) — read it
before touching `quantui/app.py`, the panel registry, or the viz backend router.

---

## Two repositories — this is the important part

QuantUI development spans **two** Git repos:

1. **This repo — [`The-Schultz-Lab/QuantUI`](https://github.com/The-Schultz-Lab/QuantUI)** (public):
   the package, tests, CI, packaging. The only thing that ships.
2. **[`NCCU-Schultz-Lab/QuantUI-development-tracking`](https://github.com/NCCU-Schultz-Lab/QuantUI-development-tracking)** (private):
   all planning — STATUS (session handoff), TODO, DECISIONS, GOTCHAS, roadmaps,
   reflections. **Never merged into this repo.**

The planning repo is where you find out *what to work on next* and *why past
decisions were made*. Start a work session by reading its
[`TODO/STATUS.md`](https://github.com/NCCU-Schultz-Lab/QuantUI-development-tracking/blob/main/TODO/STATUS.md)
and, for environment questions,
[`TODO/DEV-ENVIRONMENT.md`](https://github.com/NCCU-Schultz-Lab/QuantUI-development-tracking/blob/main/TODO/DEV-ENVIRONMENT.md).

- **Local sessions** reach the plans via the `.plans` symlink (see DEV-ENVIRONMENT).
- **Cloud sessions** must **add the planning repo to session scope** (it is a
  separate private repo — it is not cloned automatically alongside this one),
  then read/commit/push it like any repo.

---

## Environment setup

**Cloud / CI (pip, headless, CPU-only):**

```bash
python -m pip install --upgrade pip
pip install -e ".[pyscf,ase,dev,xtb,ts]"
```

This mirrors the Linux job in `.github/workflows/ci.yml`. A `SessionStart`
hook (`.claude/hooks/`) does this automatically on cloud sessions so the suite
and linters are runnable immediately. If the container's default `python` is
outside 3.9–3.11 (e.g. 3.13), that install can fail (`pyscf-properties` does
not build); create a 3.11 env instead, e.g.
`uv venv -p python3.11 .venv && . .venv/bin/activate` and install as above.

**Local (conda):** use [`local-setup/environment.yml`](local-setup/environment.yml)
(env name `quantui`). On Linux/WSL PySCF comes from conda-forge.

Supported Python: **3.9 / 3.10 / 3.11**.

---

## Test & lint

```bash
pytest -m "not network"       # what CI runs; add "and not slow and not notebook" for the fast inner loop
pytest tests/test_foo.py -n0  # one test, serial (breakpoints work; -n=auto is on by default)
pre-commit run --all-files    # file hygiene + nbstripout + black (26.5.1) + ruff (v0.16.0); mypy runs on pre-push and in CI
```

Test baseline (see STATUS): **~4120 passed / 24 skipped** for
`pytest -m "not network"` (cloud, Python 3.11, 2026-11). `network`-marked tests
need internet (live PubChem) — skip them in the cloud.

---

## What a cloud session can and cannot do

The planning docs distinguish "code-complete" from "awaiting a real-app pass."
That line **is** the cloud/local boundary:

- **Cloud can do:** code changes, `pytest -m "not network"`, ruff/black, planning
  edits, branches + PRs.
- **Local / HPC only — never mark "done" from the cloud:** Voilà live-rendering
  passes, H200/GPU (M-GPU) validation, NCShare/Apptainer deployment. Tag these as
  local follow-ups. See the planning repo's
  [`TODO/MANUAL-VERIFICATION-CHECKLIST.md`](https://github.com/NCCU-Schultz-Lab/QuantUI-development-tracking/blob/main/TODO/MANUAL-VERIFICATION-CHECKLIST.md).

---

## Working discipline

- **Branch, don't push to `main`.** CI runs on PRs into `main`.
- **Commit + push before ending a session.** Cloud containers are ephemeral;
  uncommitted work is invisible to the next machine and is lost when the
  container is reclaimed. (This has bitten the project before — see STATUS on
  stale/uncommitted branches.)
- **Commit as the project, not yourself.** Set `git config user.email nccu-schultz-lab@users.noreply.github.com` (matches `pyproject.toml`) before committing — never let a personal email into this public repo's history.
- **Keep planning and code in sync.** If a code change resolves or changes a
  roadmap item, update the planning repo in the same session.
- Do not create a PR unless asked.

---

## Key constraints (full list in copilot-instructions.md)

- **All UI logic lives in the package** — `QuantUIApp` in `quantui/app.py` plus
  its `quantui/app_*.py` companion modules; the notebook is a 3-cell launcher
  only (DEC-002).
- **Plotly embedding is `include_plotlyjs="require"` in the app (inline for
  standalone exports), never `"cdn"`** (DEC-010, guarded by
  `tests/test_code_quality.py`).
- **Analysis panels are always visible; content is swapped, not containers
  shown/hidden** (DEC-009).
- **Offline-first:** the bundled molecule library and 3Dmol.js are vendored;
  network is only for optional live structure search.
