"""QuantUI command-line interface.

Small toolkit for inspecting QuantUI state from the terminal — useful for
debugging the Voilà app from outside (e.g., when something is misbehaving
and you'd rather not open a notebook to see what happened).

Currently shipped subcommands:

* ``quantui log tail [-n N]`` — print the last N event-log entries
  (default 20). Reads ``~/.quantui/logs/event_log.jsonl`` honoring the
  ``QUANTUI_LOG_DIR`` env override.
* ``quantui gpu check`` — run QuantUI's GPU-offload detection and print
  ``(available, device-name)``. Exit code 0 when GPU is usable, 1 when
  not — handy for one-line CI / shell-script gating.
* ``quantui analytics build [-o PATH] [--open]`` — build a self-contained
  HTML analytics dashboard from ``perf_log.jsonl``. Default output:
  ``~/.quantui/dashboard.html``. Pass ``--open`` to automatically open
  the file in the default browser after writing.
* ``quantui run app [--port PORT] [--open]`` — start the Voilà student
  app. Writes ``~/.quantui/app.ipynb`` on first use (requires the
  ``[app]`` extra: ``pip install 'quantui[app]'``).
* ``quantui setup [--force]`` — write ``~/.quantui/app.ipynb`` and a
  ``quantui-app`` shell shortcut under ``~/.local/bin``.
* ``quantui submit REQUEST_JSON [REQUEST_JSON ...] [--dry-run] [--cores N]
  [--memory-gb N] [--walltime HH:MM:SS] [--email ADDR]
  [--mail-events EVENT,...] [--job-name NAME] [--depends-on REQUEST_ID]
  [--partition NAME] [--no-apptainer] [--apptainer-image PATH]`` — submit
  one or more ``CalculationRequest`` JSON files to the SLURM batch backend
  headlessly, with no interactive app/student session involved (M-CLUSTER2
  CL2.7). Resource sizing defaults to ``estimate_slurm_resources()`` —
  the same estimator the interactive app uses — unless overridden. Pass
  ``--dry-run`` to print the resolved cores/memory/walltime for each
  request without actually submitting. Respects the same
  ``QUANTUI_ENABLE_SLURM`` site gate as the interactive app.

Adding a new subcommand:

1. Write ``def _cmd_<verb>(args: argparse.Namespace) -> int:`` returning
   a POSIX-style exit code (``0`` on success).
2. Register it in ``_build_parser`` next to the existing subcommands.
3. Cover happy + empty + missing-file paths in ``tests/test_cli.py``.

The CLI deliberately avoids importing from the GUI side of the package
(``app``, ``app_builders``, ``app_visualization``) so it stays fast on
import — `quantui log tail` should not pull in ipywidgets / py3Dmol.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, cast

from quantui.calc_log import _event_path, get_recent_events


def _fmt_event(rec: dict) -> str:
    """Format one event-log record for terminal output.

    Layout: ``<timestamp>  <event:18>  <message>  [k=v k=v ...]``

    Extras (anything beyond ``timestamp`` / ``event`` / ``message``) are
    appended as ``key=value`` pairs so the line stays grep-friendly. Any
    values whose ``json.dumps`` form is uglier than ``str(value)`` (the
    common case for short strings / numbers) get the plain str rendering.
    """
    ts = str(rec.get("timestamp", ""))
    event = str(rec.get("event", "?"))
    msg = str(rec.get("message", ""))
    extras = {
        k: v for k, v in rec.items() if k not in ("timestamp", "event", "message")
    }
    extras_str = ""
    if extras:
        parts = []
        for k, v in extras.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                parts.append(f"{k}={v}")
            else:
                parts.append(f"{k}={json.dumps(v, ensure_ascii=False)}")
        extras_str = "  " + " ".join(parts)
    return f"{ts}  {event:<18}  {msg}{extras_str}"


def _cmd_log_tail(args: argparse.Namespace) -> int:
    """Print the last ``args.n`` event-log entries."""
    path = _event_path()
    if not path.exists():
        print(f"(no event log at {path})", file=sys.stderr)
        return 0
    events = get_recent_events(args.n)
    if not events:
        print("(event log is empty)", file=sys.stderr)
        return 0
    for rec in events:
        print(_fmt_event(rec))
    return 0


def _cmd_gpu_check(args: argparse.Namespace) -> int:
    """Run QuantUI's GPU detection probe and print the result.

    Returns exit code 0 when GPU offload is available, 1 when it's not —
    so ``if quantui gpu check; then ...; fi`` works in shell scripts.
    """
    from quantui.gpu_offload import is_gpu_available, is_low_fp64_device, probe_gpu

    # The detection probe is cached; clear so each CLI invocation is
    # fresh (the user may have just installed gpu4pyscf and wants to
    # confirm without restarting their shell).
    # cache_clear is forwarded from _probe_gpu's lru_cache onto this function
    # at definition time (gpu_offload.py); mypy can't see a monkey-patched
    # attribute across the module boundary.
    is_gpu_available.cache_clear()  # type: ignore[attr-defined]
    available, name, reason = probe_gpu()
    if available:
        print(f"GPU offload available: {name}")
        if is_low_fp64_device(name):
            # Available is not the same as worth using: PySCF is FP64
            # throughout, and consumer cards gate double precision to a small
            # fraction of single. Say so here rather than let the user discover
            # it as an unexplained slowdown.
            print(
                f"  note: {name} looks like a consumer/workstation GPU, whose "
                "double-precision throughput is typically 1/32–1/64 of its "
                "single-precision. Quantum-chemistry SCF is double-precision "
                "throughout, so offload here may be SLOWER than your CPU. "
                "Benchmark before relying on it; switch it off in the Status "
                "tab or with QUANTUI_DISABLE_GPU=1.",
                file=sys.stderr,
            )
        return 0
    # The reason comes straight from the probe, so this can never contradict
    # what the dispatcher actually decided.
    print("GPU offload not available", file=sys.stderr)
    if reason:
        print(f"  reason: {reason}", file=sys.stderr)
    return 1


def _is_wsl() -> bool:
    """Return True when running inside Windows Subsystem for Linux.

    Checks the cheap signal first (``WSL_DISTRO_NAME`` env var, set on
    every WSL2 distro) before falling back to a ``/proc/version`` read
    (covers WSL1 + edge cases where the env var is unset). Returns
    ``False`` on any IO error rather than raising.
    """
    import os as _os

    if _os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        with open("/proc/version", encoding="utf-8", errors="ignore") as fh:
            return "microsoft" in fh.read().lower()
    except OSError:
        return False


def _open_in_browser(path: Path) -> tuple[bool, Optional[str]]:
    """Cross-platform "open this file in the user's browser".

    On WSL, ``webbrowser.open`` ultimately calls ``xdg-open`` which fails
    on minimal Ubuntu installs ("no method available for opening...") —
    there's no native Linux browser and xdg-open doesn't know to bridge
    to the Windows host. So on WSL we prefer the WSL-aware openers in
    order: ``wslview`` (canonical xdg-open replacement, from the ``wslu``
    package), then ``explorer.exe`` (always available via WSL interop).

    Off WSL, defer to Python's stdlib ``webbrowser`` module which has the
    right per-platform handling for macOS / native Linux / Windows.

    Returns ``(success, tool_name)``. ``tool_name`` is ``None`` when no
    opener succeeded.
    """
    import subprocess

    if _is_wsl():
        # ``wslview`` accepts a Linux path directly. ``explorer.exe``
        # accepts either a Windows path OR a Linux file:// URL — but in
        # practice, passing the Linux path works through WSL interop
        # too, so we pass the path as-is to both.
        for tool in ("wslview", "explorer.exe"):
            try:
                rc = subprocess.run(
                    [tool, str(path)],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).returncode
                if rc == 0:
                    return (True, tool)
            except FileNotFoundError:
                continue
            except Exception:
                continue
        return (False, None)

    import webbrowser

    try:
        if webbrowser.open(path.as_uri()):
            return (True, "webbrowser")
    except Exception:
        pass
    return (False, None)


def _cmd_analytics_build(args: argparse.Namespace) -> int:
    """Build the HTML analytics dashboard from the perf log."""
    from quantui.analytics import build_dashboard

    out = Path(args.output) if args.output else None
    result = build_dashboard(out)
    if result is None:
        print(
            "(perf log is empty — run a calculation first)",
            file=sys.stderr,
        )
        return 0
    print(f"Wrote {result}")
    if getattr(args, "open_after", False):
        # Cross-platform open: WSL → wslview / explorer.exe; otherwise
        # stdlib webbrowser. Failure is non-fatal (the path was already
        # printed) so users can always copy-paste manually.
        opened, _tool = _open_in_browser(result)
        if not opened:
            print(
                f"(could not auto-open browser — open {result} manually)",
                file=sys.stderr,
            )
    return 0


def _cmd_run_app(args: argparse.Namespace) -> int:
    """Start the Voilà student app (lazy-provisions ~/.quantui/app.ipynb)."""
    from quantui.app_launcher import run_voila_app

    return run_voila_app(
        port=args.port,
        open_browser=args.open,
        force_notebook_refresh=args.force,
    )


def _cmd_setup(args: argparse.Namespace) -> int:
    """Write ~/.quantui/app.ipynb and a quantui-app shell shortcut."""
    from quantui.app_launcher import run_setup

    return run_setup(force=args.force)


def _cmd_submit(args: argparse.Namespace) -> int:
    """Submit one or more CalculationRequest JSON files to SLURM (M-CLUSTER2 CL2.7).

    Headless equivalent of the interactive app's Slurm batch path — no
    student session or Voilà app involved. Before this existed, the only
    way to run a batch of jobs centrally (e.g. an instructor running a
    whole class's Lab 2 series) was to hand-author sbatch scripts and
    guess at resource sizing; this wraps the exact same
    ``SlurmBackend.dispatch()`` + ``estimate_slurm_resources()`` path the
    app itself uses, so a batch caller gets the same resource-sizing
    accuracy without reinventing it.
    """
    from quantui.backends.base import CalculationRequest
    from quantui.backends.cluster_config import submit_cooldown_seconds
    from quantui.backends.dispatch import is_slurm_available, slurm_unavailable_note
    from quantui.backends.slurm_utils import estimate_slurm_resources
    from quantui.security import SecurityError

    if not args.dry_run and not is_slurm_available():
        print(f"quantui submit: {slurm_unavailable_note()}", file=sys.stderr)
        return 1

    backend = None
    if not args.dry_run:
        from quantui.backends.slurm import SlurmBackend

        backend = SlurmBackend(
            partition=args.partition,
            use_apptainer=not args.no_apptainer,
            apptainer_image=args.apptainer_image,
        )

    mail_events = args.mail_events.split(",") if args.mail_events else None
    exit_code = 0
    submitted_count = 0

    for request_path_str in args.request:
        request_path = Path(request_path_str)
        try:
            data = json.loads(request_path.read_text())
            request = CalculationRequest.from_dict(data)
        except Exception as exc:  # noqa: BLE001 — surfaced to the user below
            print(
                f"{request_path}: could not read/parse request — {exc}",
                file=sys.stderr,
            )
            exit_code = 1
            continue

        if args.dry_run:
            estimate = estimate_slurm_resources(request)
            note = ""
            if estimate.get("freq_parallel_memory_multiplier", 1) > 1:
                note = (
                    f"  [QUANTUI_FREQ_PARALLEL active: memory_gb includes a "
                    f"{estimate['freq_parallel_memory_multiplier']}x multiplier "
                    "for concurrent worker processes]"
                )
            print(
                f"{request_path}: {request.calc_type} "
                f"{request.method}/{request.basis} -> "
                f"cores={estimate['cores']} memory_gb={estimate['memory_gb']} "
                f"walltime={estimate['walltime']}  (dry run — not submitted)"
                f"{note}"
            )
            continue

        # SlurmBackend enforces a post-submit cooldown (protects the
        # interactive app from accidental rapid-fire submits). A batch of
        # N requests submitted in one invocation would otherwise trip that
        # cooldown between every pair and fail partway through a
        # legitimate scripted run — sleep it out proactively instead
        # (this is exactly CL2.7's motivating case: an instructor running
        # a whole class's worth of jobs, e.g. Lab 2's 30, from one script).
        if submitted_count > 0:
            cooldown = submit_cooldown_seconds()
            if cooldown > 0:
                time.sleep(cooldown)

        assert backend is not None  # not args.dry_run, guarded above
        try:
            request_id = backend.dispatch(
                request,
                cores=args.cores,
                memory_gb=args.memory_gb,
                walltime=args.walltime,
                depends_on=args.depends_on,
                email=args.email,
                mail_events=mail_events,
                job_name=args.job_name,
            )
        except SecurityError as exc:
            print(f"{request_path}: rejected — {exc}", file=sys.stderr)
            exit_code = 1
            continue
        except RuntimeError as exc:
            print(f"{request_path}: submission failed — {exc}", file=sys.stderr)
            exit_code = 1
            continue

        submitted_count += 1
        record = backend.registry.load(request_id)
        slurm_job_id = record.slurm_job_id if record else None
        staging = record.staging_path if record else "?"
        print(
            f"{request_path}: submitted request_id={request_id} "
            f"slurm_job_id={slurm_job_id or '?'} staging={staging}"
        )

    return exit_code


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantui",
        description="QuantUI command-line toolkit.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    log_parser = sub.add_parser("log", help="Inspect QuantUI's event log.")
    log_sub = log_parser.add_subparsers(dest="log_command", required=True)
    tail = log_sub.add_parser(
        "tail",
        help="Print the last N events from event_log.jsonl.",
    )
    tail.add_argument(
        "-n",
        type=int,
        default=20,
        metavar="N",
        help="Number of most-recent events to print (default: 20).",
    )
    tail.set_defaults(func=_cmd_log_tail)

    gpu_parser = sub.add_parser("gpu", help="GPU offload utilities.")
    gpu_sub = gpu_parser.add_subparsers(dest="gpu_command", required=True)
    gpu_check = gpu_sub.add_parser(
        "check",
        help="Run QuantUI's GPU-offload detection probe.",
    )
    gpu_check.set_defaults(func=_cmd_gpu_check)

    analytics_parser = sub.add_parser(
        "analytics", help="Build usage analytics reports."
    )
    analytics_sub = analytics_parser.add_subparsers(
        dest="analytics_command", required=True
    )
    analytics_build = analytics_sub.add_parser(
        "build",
        help="Build the HTML analytics dashboard from perf_log.jsonl.",
    )
    analytics_build.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="Output HTML path (default: ~/.quantui/dashboard.html).",
    )
    analytics_build.add_argument(
        "--open",
        dest="open_after",
        action="store_true",
        help=(
            "After writing, open the dashboard in the default browser "
            "(via webbrowser.open). Best-effort — falls back to printing "
            "the path on headless systems."
        ),
    )
    analytics_build.set_defaults(func=_cmd_analytics_build)

    setup_parser = sub.add_parser(
        "setup",
        help="Write ~/.quantui/app.ipynb and a quantui-app shell shortcut.",
    )
    setup_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing launcher notebook or shell script.",
    )
    setup_parser.set_defaults(func=_cmd_setup)

    run_parser = sub.add_parser("run", help="Run QuantUI services.")
    run_sub = run_parser.add_subparsers(dest="run_command", required=True)
    run_app = run_sub.add_parser(
        "app",
        help="Start the Voilà student app (pip install 'quantui[app]').",
    )
    run_app.add_argument(
        "--port",
        type=int,
        default=8867,
        metavar="PORT",
        help="TCP port for Voilà (default: 8867, matches native launchers).",
    )
    run_app.add_argument(
        "--open",
        action="store_true",
        help="Open http://localhost:PORT in the default browser after startup.",
    )
    run_app.add_argument(
        "--force",
        action="store_true",
        help="Regenerate ~/.quantui/app.ipynb before starting.",
    )
    run_app.set_defaults(func=_cmd_run_app)

    submit_parser = sub.add_parser(
        "submit",
        help=(
            "Submit CalculationRequest JSON file(s) to the SLURM batch "
            "backend headlessly (M-CLUSTER2 CL2.7) — no interactive app."
        ),
    )
    submit_parser.add_argument(
        "request",
        nargs="+",
        metavar="REQUEST_JSON",
        help="Path(s) to a CalculationRequest JSON file.",
    )
    submit_parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help=(
            "Print the resolved cores/memory/walltime estimate for each "
            "request without submitting to SLURM."
        ),
    )
    submit_parser.add_argument(
        "--cores",
        type=int,
        default=None,
        metavar="N",
        help="Override the estimated core count (default: estimate_slurm_resources()).",
    )
    submit_parser.add_argument(
        "--memory-gb",
        dest="memory_gb",
        type=int,
        default=None,
        metavar="N",
        help="Override the estimated memory in GB.",
    )
    submit_parser.add_argument(
        "--walltime",
        type=str,
        default=None,
        metavar="HH:MM:SS",
        help="Override the estimated walltime.",
    )
    submit_parser.add_argument(
        "--email",
        type=str,
        default=None,
        metavar="ADDR",
        help="Email address for SLURM job notifications.",
    )
    submit_parser.add_argument(
        "--mail-events",
        dest="mail_events",
        type=str,
        default=None,
        metavar="EVENT,EVENT,...",
        help=(
            "Comma-separated SLURM mail events (e.g. END,FAIL). "
            "Defaults to QuantUI's standard set when --email is given."
        ),
    )
    submit_parser.add_argument(
        "--job-name",
        dest="job_name",
        type=str,
        default=None,
        metavar="NAME",
        help="SLURM job name (default: derived from the molecule + method).",
    )
    submit_parser.add_argument(
        "--depends-on",
        dest="depends_on",
        type=str,
        default=None,
        metavar="REQUEST_ID",
        help="Make this job depend on another already-submitted request's completion.",
    )
    submit_parser.add_argument(
        "--partition",
        type=str,
        default=None,
        metavar="NAME",
        help="SLURM partition (default: site-configured DEFAULT_PARTITION).",
    )
    submit_parser.add_argument(
        "--no-apptainer",
        dest="no_apptainer",
        action="store_true",
        help="Run the worker with the current Python instead of inside Apptainer.",
    )
    submit_parser.add_argument(
        "--apptainer-image",
        dest="apptainer_image",
        type=str,
        default=None,
        metavar="PATH",
        help="Override the Apptainer image path (default: site-configured).",
    )
    submit_parser.set_defaults(func=_cmd_submit)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    # args.func is set per-subcommand via set_defaults(func=_cmd_xxx)
    # (argparse.Namespace is untyped, so mypy sees Any here); every _cmd_*
    # handler returns an int exit code by convention.
    return cast(int, args.func(args))


if __name__ == "__main__":
    sys.exit(main())
