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
from pathlib import Path
from typing import Optional, Sequence

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
    is_gpu_available.cache_clear()
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

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
