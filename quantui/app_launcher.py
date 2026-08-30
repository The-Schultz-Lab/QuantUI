"""Voilà app launcher helpers for ``quantui run app`` and ``quantui setup``.

The student-facing UI is a thin notebook that calls ``QuantUIApp().display()``.
Pip installs do not ship the repo's ``notebooks/`` tree, so the CLI writes an
equivalent launcher notebook to ``~/.quantui/app.ipynb`` on first use (or when
``quantui setup`` runs).

The module intentionally does not import ``quantui.app`` — only ``voila`` is
required at launch time via the ``[app]`` extra.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

DEFAULT_APP_PORT = 8867
APP_NOTEBOOK_NAME = "app.ipynb"
HOME_LAUNCHER_NOTEBOOK_NAME = "QuantUI.ipynb"
LAUNCHER_SCRIPT_NAME = "quantui-app"

# Minimal nbformat v4 notebook — mirrors notebooks/molecule_computations.ipynb
# (display cell only; no repo-root sys.path hack needed for installed packages).
_APP_NOTEBOOK: dict = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# QuantUI\n"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"tags": ["remove-input"]},
            "outputs": [],
            "source": [
                "from quantui.app import QuantUIApp\n",
                "\n",
                "QuantUIApp().display()\n",
            ],
        },
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def quantui_home() -> Path:
    """Return the QuantUI user config directory (``~/.quantui`` by default)."""
    override = os.environ.get("QUANTUI_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".quantui"


def app_notebook_path() -> Path:
    """Path to the generated Voilà launcher notebook."""
    return quantui_home() / APP_NOTEBOOK_NAME


def home_launcher_notebook_path() -> Path:
    """Path to a JupyterLab-visible launcher notebook in the user's home."""
    return Path.home() / HOME_LAUNCHER_NOTEBOOK_NAME


def is_apptainer_runtime() -> bool:
    """Return True when running inside Apptainer/Singularity."""
    for key in (
        "APPTAINER_CONTAINER",
        "APPTAINER_NAME",
        "SINGULARITY_CONTAINER",
        "SINGULARITY_NAME",
    ):
        if os.environ.get(key):
            return True
    return Path("/.singularity.d").is_dir()


def is_jupyter_server_context() -> bool:
    """Return True when a Jupyter server session is active in this environment."""
    for key in (
        "JUPYTER_SERVER_URL",
        "JUPYTERHUB_SERVICE_URL",
        "JUPYTERHUB_USER",
        "JPY_SESSION_NAME",
    ):
        if os.environ.get(key):
            return True
    return False


def is_hpc_jupyterlab_session() -> bool:
    """Heuristic for NCShare-style Apptainer + JupyterLab interactive sessions."""
    return is_apptainer_runtime() and is_jupyter_server_context()


def launcher_bin_dir() -> Path:
    """Preferred directory for the ``quantui-app`` shell wrapper."""
    xdg = os.environ.get("XDG_BIN_HOME")
    if xdg:
        return Path(xdg).expanduser()
    return Path.home() / ".local" / "bin"


def launcher_script_path() -> Path:
    return launcher_bin_dir() / LAUNCHER_SCRIPT_NAME


def voila_executable() -> Optional[str]:
    """Return the ``voila`` executable path, or ``None`` if not installed."""
    return shutil.which("voila")


def voila_missing_message() -> str:
    return (
        "Voilà is not installed or not on PATH.\n"
        "Install the app extra, then retry:\n"
        "  pip install 'quantui[app]'\n"
        "  # or, from a dev clone:\n"
        "  pip install -e '.[app]'"
    )


def ensure_app_notebook(*, force: bool = False) -> Path:
    """Write ``~/.quantui/app.ipynb`` when missing (or when *force* is True)."""
    home = quantui_home()
    home.mkdir(parents=True, exist_ok=True)
    path = app_notebook_path()
    if path.exists() and not force:
        return path
    path.write_text(
        json.dumps(_APP_NOTEBOOK, indent=1) + "\n",
        encoding="utf-8",
    )
    return path


def ensure_home_launcher_notebook(*, force: bool = False) -> Path:
    """Write ``~/QuantUI.ipynb`` for JupyterLab file-browser visibility."""
    path = home_launcher_notebook_path()
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_APP_NOTEBOOK, indent=1) + "\n",
        encoding="utf-8",
    )
    return path


def hpc_jupyterlab_run_app_message() -> str:
    """Explain why ``quantui run app`` is the wrong entry point on NCShare."""
    nb = home_launcher_notebook_path()
    return (
        "QuantUI detected an HPC JupyterLab session (Apptainer + Jupyter).\n"
        "\n"
        "``quantui run app`` starts a standalone Voilà server on a separate\n"
        "port. Cluster portals (including NCShare) proxy only the Jupyter\n"
        "connection, so that port is not reachable from your browser.\n"
        "\n"
        "Use one of these instead:\n"
        f'  1. Open {nb} in JupyterLab and click "Render with Voilà"\n'
        "  2. Run the first cell in any notebook:\n"
        "       from quantui.app import QuantUIApp\n"
        "       QuantUIApp().display()\n"
        "\n"
        "Run ``quantui setup`` once to create ~/QuantUI.ipynb if it is missing."
    )


def hpc_jupyterlab_setup_message(home_nb: Path) -> str:
    """Post-setup instructions for NCShare-style JupyterLab sessions."""
    return (
        "HPC JupyterLab session detected (Apptainer + Jupyter).\n"
        "\n"
        "On NCShare and similar clusters, do NOT use ``quantui run app`` — the\n"
        "browser cannot reach a second Voilà port.\n"
        "\n"
        "Launch QuantUI from JupyterLab instead:\n"
        f'  • Open {home_nb} and click "Render with Voilà" (clean student view)\n'
        "  • Or run the first cell in any notebook:\n"
        "      from quantui.app import QuantUIApp\n"
        "      QuantUIApp().display()\n"
    )


def build_voila_argv(
    notebook: Path,
    *,
    port: int = DEFAULT_APP_PORT,
    no_browser: bool = True,
) -> List[str]:
    """Return a ``voila`` argv list matching the native launchers."""
    argv = [
        "voila",
        str(notebook),
        f"--port={port}",
        "--ServerApp.disable_check_xsrf=True",
    ]
    if no_browser:
        argv.append("--no-browser")
    return argv


def write_launcher_script(*, force: bool = False) -> Path:
    """Write ``~/.local/bin/quantui-app`` (or ``$XDG_BIN_HOME/quantui-app``)."""
    bindir = launcher_bin_dir()
    bindir.mkdir(parents=True, exist_ok=True)
    path = launcher_script_path()
    if path.exists() and not force:
        return path
    content = (
        "#!/usr/bin/env bash\n"
        "# QuantUI Voilà launcher — generated by ``quantui setup``.\n"
        "set -eu\n"
        'exec quantui run app "$@"\n'
    )
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def run_voila_app(
    *,
    port: int = DEFAULT_APP_PORT,
    open_browser: bool = False,
    force_notebook_refresh: bool = False,
) -> int:
    """Provision the launcher notebook and start Voilà.

    Replaces the current process with ``voila`` when *open_browser* is False
    (the common case). When *open_browser* is True, Voilà runs in a child
    process so the CLI can open the URL after a short bind delay.
    """
    if is_hpc_jupyterlab_session():
        print(hpc_jupyterlab_run_app_message(), file=sys.stderr)
        return 1

    voila = voila_executable()
    if voila is None:
        print(voila_missing_message(), file=sys.stderr)
        return 1

    notebook = ensure_app_notebook(force=force_notebook_refresh)
    argv = build_voila_argv(notebook, port=port, no_browser=True)
    argv[0] = voila

    url = f"http://localhost:{port}"
    print(f"Starting QuantUI at {url}")
    print(f"Notebook: {notebook}")

    if not open_browser:
        print("Press Ctrl-C to stop.")
        os.execvp(voila, argv)

    proc = subprocess.Popen(argv)
    time.sleep(4)
    _open_url_best_effort(url)
    print("Press Ctrl-C to stop.")
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 130


def _open_url_best_effort(url: str) -> None:
    """Open *url* in the user's browser (best-effort, never raises)."""
    import subprocess

    if os.environ.get("WSL_DISTRO_NAME") or _is_wsl():
        for tool in ("wslview", "explorer.exe"):
            try:
                if (
                    subprocess.run(
                        [tool, url],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    ).returncode
                    == 0
                ):
                    return
            except (FileNotFoundError, OSError):
                continue
    else:
        import webbrowser

        try:
            if webbrowser.open(url):
                return
        except Exception:
            pass
    print(f"(could not auto-open browser — open {url} manually)", file=sys.stderr)


def _is_wsl() -> bool:
    try:
        with open("/proc/version", encoding="utf-8", errors="ignore") as fh:
            return "microsoft" in fh.read().lower()
    except OSError:
        return False


def run_setup(*, force: bool = False) -> int:
    """Write the launcher notebook and optional ``quantui-app`` shell script."""
    notebook = ensure_app_notebook(force=force)
    script = write_launcher_script(force=force)
    bindir = launcher_bin_dir()
    hpc = is_hpc_jupyterlab_session()
    home_nb: Optional[Path] = None
    if hpc:
        home_nb = ensure_home_launcher_notebook(force=force)

    print(f"Wrote launcher notebook: {notebook}")
    if home_nb is not None:
        print(f"Wrote home launcher:     {home_nb}")
    print(f"Wrote shell shortcut:    {script}")
    print()
    if hpc and home_nb is not None:
        print(hpc_jupyterlab_setup_message(home_nb))
    else:
        print("Run the app with either:")
        print("  quantui run app")
        print(f"  {script}")
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if str(bindir) not in path_entries:
            print()
            print(f"Add {bindir} to your PATH to run ``quantui-app`` from anywhere:")
            print(f'  export PATH="{bindir}:$PATH"')
    return 0


__all__ = [
    "DEFAULT_APP_PORT",
    "app_notebook_path",
    "build_voila_argv",
    "ensure_app_notebook",
    "ensure_home_launcher_notebook",
    "home_launcher_notebook_path",
    "hpc_jupyterlab_run_app_message",
    "hpc_jupyterlab_setup_message",
    "is_apptainer_runtime",
    "is_hpc_jupyterlab_session",
    "is_jupyter_server_context",
    "launcher_script_path",
    "quantui_home",
    "run_setup",
    "run_voila_app",
    "voila_executable",
    "voila_missing_message",
    "write_launcher_script",
]
