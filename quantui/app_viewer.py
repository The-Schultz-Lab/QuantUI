"""Viewer mode — History + Analysis only, pointed at a results folder.

``QuantUIApp(viewer=True)`` builds the normal app but shows only the History
and Analysis tabs, skips calc-related startup work (GPU probe, resume offers,
SLURM checks), and adds a folder bar so the user can point the History browser
at any QuantUI results folder. No quantum engine is needed to browse results.

The results folder is selected through ``QUANTUI_RESULTS_DIR`` — the same seam
``results_storage._default_results_dir`` already reads on every call (and the
Apptainer image sets), so every History/Compare/Files code path follows it
without being threaded a new argument.

In a browser (WebAssembly/Pyodide) build there is no local disk to point at,
so the bar also offers a ``.zip`` upload: the archive is unpacked into the
browser's in-memory filesystem and opened like any other folder.
"""

from __future__ import annotations

import html as _html
import io
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Optional

import ipywidgets as widgets

from . import theme as _theme

RESULTS_DIR_ENV = "QUANTUI_RESULTS_DIR"
IN_BROWSER = sys.platform == "emscripten"
UPLOAD_ROOT = Path("uploaded_results")


def set_results_dir(path: Path) -> None:
    """Point every results reader at *path* (process-wide)."""
    os.environ[RESULTS_DIR_ENV] = str(path)


def resolve_results_folder(text: str) -> tuple[Optional[Path], str]:
    """Validate a user-typed folder path.

    Returns ``(path, "")`` on success or ``(None, reason)``. A folder holding
    ``result.json`` directly is one result, not a results folder — its parent
    is used instead so a user who picks a single run still sees it listed.
    """
    raw = (text or "").strip().strip('"').strip("'")
    if not raw:
        return None, "Enter a folder path."
    path = Path(raw).expanduser()
    try:
        path = path.resolve()
    except OSError as exc:
        return None, f"Could not read that path: {exc}"
    if not path.is_dir():
        return None, "That folder does not exist."
    if (path / "result.json").is_file():
        path = path.parent
    return path, ""


def build_viewer_folder_bar(app: Any, *, layout_fn: Any) -> None:
    """Build the results-folder picker shown above the tabs in viewer mode."""
    from quantui.results_storage import _default_results_dir

    app._viewer_folder_txt = widgets.Text(
        value=str(_default_results_dir().expanduser().resolve()),
        placeholder="Path to a QuantUI results folder",
        description="Results folder:",
        style={"description_width": "110px"},
        layout=layout_fn(width="640px"),
    )
    app._viewer_open_btn = widgets.Button(
        description="Open",
        icon="folder-open",
        button_style="primary",
        tooltip="Load the calculations saved in this folder",
        layout=layout_fn(width="90px"),
    )
    app._viewer_status_html = widgets.HTML("")
    app._viewer_open_btn.on_click(lambda _btn: on_open_folder(app))
    children = [app._viewer_folder_txt, app._viewer_open_btn]
    if IN_BROWSER:
        app._viewer_upload = widgets.FileUpload(
            accept=".zip",
            multiple=False,
            description="Upload .zip",
            tooltip="Upload a zipped QuantUI results folder",
            layout=layout_fn(width="130px"),
        )
        app._viewer_upload.observe(
            lambda change: on_upload_zip(app, change["new"]), names="value"
        )
        children.append(app._viewer_upload)
        # No local server to stop — Exit would only kill the browser kernel.
        app._exit_btn.layout.display = "none"
    children.append(app._viewer_status_html)
    app._viewer_folder_bar = widgets.HBox(
        children,
        layout=layout_fn(align_items="center", gap="8px", margin="0 0 8px"),
    )


def extract_results_zip(data: bytes, name: str, root: Path = UPLOAD_ROOT) -> Path:
    """Unpack a zipped results folder under *root*; return the folder to open.

    Rejects members that would escape the destination (``..`` or absolute
    paths). When the archive wraps everything in one top-level folder, that
    folder is returned so History sees the result directories directly.
    """
    dest = (root / Path(name).stem).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            target = (dest / member).resolve()
            if dest != target and dest not in target.parents:
                raise ValueError(f"Unsafe path in archive: {member}")
        zf.extractall(dest)
    entries = [p for p in dest.iterdir() if not p.name.startswith(("__MACOSX", "."))]
    if (
        len(entries) == 1
        and entries[0].is_dir()
        and not (entries[0] / "result.json").is_file()
    ):
        return entries[0]
    return dest


def on_upload_zip(app: Any, value: Any) -> None:
    """Unpack an uploaded results zip and open it."""
    if not value:
        return
    item = value[0]
    try:
        folder = extract_results_zip(bytes(item["content"]), item["name"])
    except (zipfile.BadZipFile, ValueError, OSError) as exc:
        _set_status(app, f"Could not read {item['name']}: {exc}", error=True)
        return
    app._viewer_folder_txt.value = str(folder)
    on_open_folder(app)


def on_open_folder(app: Any) -> None:
    """Switch the History browser to the folder typed in the folder bar."""
    path, reason = resolve_results_folder(app._viewer_folder_txt.value)
    if path is None:
        _set_status(app, reason, error=True)
        return
    set_results_dir(path)
    app._viewer_folder_txt.value = str(path)
    app._deactivate_all_ana_panels()
    app._refresh_results_browser()
    from quantui.results_storage import list_results

    n = len(list_results(path))
    _set_status(
        app,
        (
            f"{n} calculation{'s' if n != 1 else ''} found."
            if n
            else "No QuantUI results found in this folder."
        ),
        error=not n,
    )
    app.root_tab.selected_index = app._tab_index("history")


def _set_status(app: Any, text: str, *, error: bool) -> None:
    color = _theme.css.ACCENT_ERROR if error else _theme.css.TEXT_SECONDARY
    app._viewer_status_html.value = (
        f'<span style="color:{color};font-size:12px">{_html.escape(text)}</span>'
    )
