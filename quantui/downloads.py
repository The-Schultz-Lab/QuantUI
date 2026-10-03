"""Browser downloads for files QuantUI writes on the server (DEC-023 Tier-1 #1).

Every export lands on the machine running the kernel. On a laptop that is
the student's own disk, but on Voilà/OnDemand/NCShare it is a remote node
the student cannot browse. A download link carries the bytes over the
widget channel as a ``data:`` URI with the HTML ``download`` attribute, so
it works wherever the app's widgets work: no file-server route, no extra
port, no Jupyter file browser.

Data URIs grow by a third (base64) and travel through the kernel's comm, so
large files are refused with a pointer to a better route instead of
freezing the browser.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Optional, Union

#: Largest file offered as an inline download (bytes on disk).
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024

_MIME = {
    ".png": "image/png",
    ".html": "text/html",
    ".json": "application/json",
    ".csv": "text/csv",
    ".txt": "text/plain",
    ".log": "text/plain",
    ".xyz": "chemical/x-xyz",
    ".mol": "chemical/x-mdl-molfile",
    ".sdf": "chemical/x-mdl-sdfile",
    ".pdb": "chemical/x-pdb",
    ".cube": "chemical/x-cube",
    ".molden": "chemical/x-molden",
    ".py": "text/x-python",
    ".zip": "application/zip",
}


def download_link_html(path: Union[str, Path], label: Optional[str] = None) -> str:
    """An ``<a download>`` link carrying *path*'s bytes, or a short notice.

    Returns a notice (never raises) for a missing file or one larger than
    :data:`MAX_DOWNLOAD_BYTES`.
    """
    p = Path(path)
    name = html.escape(p.name)
    try:
        size = p.stat().st_size
    except OSError:
        return f'<span style="color:#b22">{name} not found.</span>'
    if size > MAX_DOWNLOAD_BYTES:
        mb = size / (1024 * 1024)
        return (
            f'<span style="color:#92400e">{name} is {mb:.0f} MB — too large '
            f"for a browser download link (limit "
            f"{MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB). Copy it from the "
            "result folder instead.</span>"
        )
    mime = _MIME.get(p.suffix.lower(), "application/octet-stream")
    payload = base64.b64encode(p.read_bytes()).decode("ascii")
    text = html.escape(label) if label else f"⬇ Download {name}"
    return (
        f'<a download="{name}" href="data:{mime};base64,{payload}" '
        'style="font-weight:600;text-decoration:underline">'
        f"{text}</a>"
    )


def saved_with_download_html(path: Union[str, Path]) -> str:
    """'Saved: <name>' status plus a download link, for HTML status widgets."""
    p = Path(path)
    return (
        f'<span style="color:#2a7">Saved: {html.escape(p.name)}</span> '
        + download_link_html(p)
    )


def offer_download(app: object, path: Union[str, Path]) -> None:
    """Show a download link for *path* in the Results-tab download slot."""
    slot = getattr(app, "_download_html", None)
    if slot is None:
        return
    try:
        slot.value = download_link_html(path)
    except Exception as exc:  # noqa: BLE001 — a link must never break an export
        slot.value = f'<span style="color:#b22">Download link failed: {exc}</span>'
