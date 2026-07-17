#!/usr/bin/env python
"""Offline-startup probe for QuantUI — does the app start with NO network?

Simulates "Python is offline" by blocking (and logging) every non-local socket
connect / DNS lookup, then imports the app, constructs it, and calls
``display()`` — exactly what the notebook's launch cell does. Reports whether
startup completed, how long it took, and **every network call the app attempted
during startup** (there should be none; any is a candidate for an offline
hang).

This catches the class of "app doesn't launch offline" bug where some import or
constructor phones home (a CDN, a version check, a structure-DB warmup) and
blocks waiting for a connect timeout when there's no network. It runs without a
browser or a server, so it's fast and deterministic.

NOTE: this checks the *app*. The native launchers (``launch-native.bat`` etc.)
also run ``pip install -e .`` when ``pyproject.toml`` changes — that step
fetches build deps from PyPI and will hang/fail offline, so the launchers run it
fail-fast + non-fatal. If "doesn't start up offline" recurs, check BOTH: this
probe (app side) and the launcher's reinstall step.

Usage:
    python scripts/test_offline_startup.py
Exit 0 = startup completed offline with no network calls; 1 = it didn't.
"""

from __future__ import annotations

import contextlib
import io
import socket
import sys
import time
import traceback

_LOCAL = {"127.0.0.1", "::1", "localhost", "0.0.0.0", ""}
_attempts: list[tuple[str, str]] = []

_orig_connect = socket.socket.connect
_orig_getaddrinfo = socket.getaddrinfo


def _host_of(address: object) -> str:
    if isinstance(address, (tuple, list)) and address:
        return str(address[0])
    return str(address)


def _blocked_connect(self, address):  # type: ignore[no-untyped-def]
    if _host_of(address) in _LOCAL:
        return _orig_connect(self, address)
    _attempts.append(("connect", str(address)))
    raise OSError(f"SIMULATED OFFLINE: connect blocked -> {address!r}")


def _blocked_getaddrinfo(host, *args, **kwargs):  # type: ignore[no-untyped-def]
    if str(host) in _LOCAL:
        return _orig_getaddrinfo(host, *args, **kwargs)
    _attempts.append(("dns", str(host)))
    raise socket.gaierror(f"SIMULATED OFFLINE: DNS blocked -> {host!r}")


def main() -> int:
    socket.socket.connect = _blocked_connect  # type: ignore[method-assign]
    socket.getaddrinfo = _blocked_getaddrinfo  # type: ignore[assignment]

    print("=== QuantUI OFFLINE-STARTUP PROBE (Python network blocked) ===")
    t0 = time.time()
    ok = True
    try:
        from quantui.app import QuantUIApp

        t_import = time.time() - t0
        t1 = time.time()
        # display() with no Jupyter frontend repr-prints the whole widget tree;
        # silence it so the probe output stays readable (we only care that it
        # runs without a network call / hang).
        with contextlib.redirect_stdout(io.StringIO()):
            QuantUIApp().display()
        t_app = time.time() - t1
        total = time.time() - t0
        print(f"import quantui.app : {t_import:.2f}s")
        print(f"construct+display  : {t_app:.2f}s")
        print(f"STARTUP COMPLETED OFFLINE in {total:.2f}s")
    except Exception:
        ok = False
        print(f"STARTUP FAILED after {time.time() - t0:.2f}s:")
        print(traceback.format_exc())
    finally:
        socket.socket.connect = _orig_connect  # type: ignore[method-assign]
        socket.getaddrinfo = _orig_getaddrinfo  # type: ignore[assignment]

    print(f"\nnetwork calls attempted during startup: {len(_attempts)}")
    for kind, target in _attempts:
        print(f"    {kind} -> {target}")
    if _attempts:
        print(
            "\nWARNING: startup attempted network I/O — each call would hang or "
            "fail offline. Make it lazy (on user action) or bundle the resource."
        )

    passed = ok and not _attempts
    print("\nRESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
