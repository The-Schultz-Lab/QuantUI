"""CI smoke test: start the installed viewer the way a shortcut does and
check that it serves a page.

    python smoke_launch.py PATH/TO/quantui[.exe]

Run from a shell where the installed environment is NOT activated, so this
catches anything that only works when its folders are on PATH (the first
Windows build failed exactly that way: "Voila is not installed or not on
PATH" from a Start-menu shortcut).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def main(quantui: str) -> int:
    port = _free_port()  # never a server left over from an earlier run
    proc = subprocess.Popen([quantui, "view", "--no-browser", "--port", str(port)])
    deadline = time.monotonic() + 120
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                print(f"quantui view exited early with code {proc.returncode}")
                return 1
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}", timeout=5) as r:
                    if r.status == 200:
                        print("viewer served HTTP 200")
                        break
            except OSError:
                time.sleep(2)
        else:
            print("viewer did not answer within 120 s")
            return 1
    finally:
        if os.name == "nt":
            # TerminateProcess runs no handlers, so Voila (a child) would
            # outlive it; kill the whole tree. Real Windows use is covered
            # differently: closing the console window stops every process
            # attached to it, and the app's Exit button stops Voila itself.
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                capture_output=True,
            )
        else:
            proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    if os.name == "nt":
        return 0
    # macOS/Linux: SIGTERM to quantui must stop Voila too (the launcher
    # forwards it), not leave it serving.
    time.sleep(3)
    if _port_open(port):
        print(f"Voila still serving on port {port} after quantui stopped")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
