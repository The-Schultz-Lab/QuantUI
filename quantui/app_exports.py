"""Export helpers used by QuantUIApp."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, cast

from .results_storage import _safe_name


def general_figures_dir() -> Path:
    """Folder for viewer/plot PNGs that are not tied to a calculation run."""
    dest = Path.home() / ".quantui" / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def export_destination(
    app: Any,
    category: str,
    *name_parts: str,
    suffix: str,
    general_if_no_result: bool = False,
    timestamp: bool = False,
) -> Path:
    """The one place that decides where a new export lands and what it's
    called (M-EXPORT2 EXP2.3).

    Before this, each exporter picked its own destination and built its own
    filename inline (``on_export_xyz`` / ``on_export_mol`` / ``on_export_pdb``
    below all repeat the same three-line pattern), so "where did it save?"
    had a different answer depending on which button was pressed. New
    exporters should call this instead of repeating that pattern; the three
    existing structure exporters are left as-is deliberately — they already
    work, and retrofitting working export paths carries real regression risk
    for no user-facing benefit. This is about not repeating the inconsistency
    as the export surface grows (EXP2.1, and whatever comes after it).

    Every part of ``name_parts`` is sanitised (:func:`_safe_name`) and joined
    with underscores, so a caller passes meaningful pieces (formula, a
    geometry label, ...) instead of building a filename by hand.

    Files land next to the calculation's own results (``app._last_result_dir``)
    so everything about one run stays in one folder. Unlike the existing
    exporters (which fall back to the current working directory when no
    result folder exists yet), this raises — silently writing outside the
    result folder is a worse default for anything added from here on.

    Args:
        app: the running QuantUIApp (only ``_last_result_dir`` is read).
        category: a short, human-readable export kind, used only in the
            error message (e.g. ``"reorg geometry"``).
        *name_parts: filename-stem pieces, sanitised and joined with ``"_"``.
        suffix: file extension including the leading dot (e.g. ``".xyz"``).

    Returns:
        The full destination path (parent directory already exists, since it
        is always an existing result directory).

    Raises:
        ValueError: no result directory is available yet.
    """
    result_dir = getattr(app, "_last_result_dir", None)
    using_general = result_dir is None or not isinstance(result_dir, Path)
    if using_general:
        if general_if_no_result:
            result_dir = general_figures_dir()
        else:
            raise ValueError(
                f"No result folder yet — run a calculation before exporting {category}."
            )
    parts = list(name_parts)
    if timestamp and using_general:
        parts.append(datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f"))
    stem = "_".join(_safe_name(str(p)) for p in parts if p)
    # result_dir is narrowed to Path by the isinstance check above, but mypy's
    # Path.__truediv__ overload resolution still infers Any from an
    # originally-Any-typed (getattr on `app: Any`) operand — verified in
    # isolation; the isinstance check is the real, working type guard.
    return cast(Path, result_dir / f"{stem}{suffix}")


def on_export(app: Any, btn: Any) -> None:
    """Export a standalone Python calculation script."""
    if app._molecule is None:
        app.export_status.value = "Load a molecule first."
        return
    try:
        from quantui import PySCFCalculation

        calc = PySCFCalculation(
            app._molecule,
            method=app.method_dd.value,
            basis=app.basis_dd.value,
        )
        # M11 audit fix (2026-07-14): the basis set is embedded verbatim in
        # the filename (e.g. "6-31G*.py"), and "*" is invalid in a Windows
        # filename — this export silently failed there. _safe_name (already
        # used by results_storage for the same purpose) replaces anything
        # that isn't alphanumeric/underscore/hyphen with "x".
        fname = (
            f"{_safe_name(app._molecule.get_formula())}"
            f"_{_safe_name(app.method_dd.value)}_{_safe_name(app.basis_dd.value)}.py"
        )
        calc.generate_calculation_script(Path(fname))
        app.export_status.value = f"Saved: {fname}"
    except Exception as exc:
        app.export_status.value = f"Error: {exc}"


def on_export_xyz(app: Any, btn: Any) -> None:
    """Export molecule geometry to an XYZ file.

    Provenance (M-EXPORT2 EXP2.4): the comment line carries charge and
    multiplicity alongside method/basis, matching
    :func:`on_export_reorg_geometries`'s format — before this fix the two
    XYZ exporters disagreed (reorg had charge/multiplicity, this one didn't),
    and charge/multiplicity is exactly the kind of thing unrecoverable from a
    bare geometry once it's been handed off.
    """
    if app._molecule is None:
        app.struct_export_status.value = "Load a molecule first."
        return
    try:
        mol, method, basis = export_molecule_and_label(app)
        fname = f"{_safe_name(mol.get_formula())}_{_safe_name(method)}_{_safe_name(basis)}.xyz"
        xyz_body = mol.to_xyz_string()
        comment = (
            f"{mol.get_formula()}  charge={mol.charge} multiplicity={mol.multiplicity}  "
            f"{method}/{basis}"
        )
        full_xyz = f"{len(mol.atoms)}\n{comment}\n{xyz_body}\n"
        dest = (app._last_result_dir / fname) if app._last_result_dir else Path(fname)
        dest.write_text(full_xyz, encoding="utf-8")
        app.struct_export_status.value = f"Saved: {dest}"
    except Exception as exc:
        app.struct_export_status.value = f"Error: {exc}"


def on_export_mol(app: Any, btn: Any) -> None:
    """Export molecule geometry to a MOL file via RDKit."""
    if app._molecule is None:
        app.struct_export_status.value = "Load a molecule first."
        return
    try:
        from rdkit import Chem

        mol, method, basis = export_molecule_and_label(app)
        fname = f"{_safe_name(mol.get_formula())}_{_safe_name(method)}_{_safe_name(basis)}.mol"
        rdmol = molecule_to_rdkit(mol)
        if rdmol is None:
            app.struct_export_status.value = "RDKit could not parse the structure."
            return
        mol_block = Chem.MolToMolBlock(rdmol)
        dest = (app._last_result_dir / fname) if app._last_result_dir else Path(fname)
        dest.write_text(mol_block, encoding="utf-8")
        app.struct_export_status.value = f"Saved: {dest}"
    except Exception as exc:
        app.struct_export_status.value = f"Error: {exc}"


def on_export_pdb(app: Any, btn: Any) -> None:
    """Export molecule geometry to a PDB file via RDKit."""
    if app._molecule is None:
        app.struct_export_status.value = "Load a molecule first."
        return
    try:
        from rdkit import Chem

        mol, method, basis = export_molecule_and_label(app)
        fname = f"{_safe_name(mol.get_formula())}_{_safe_name(method)}_{_safe_name(basis)}.pdb"
        rdmol = molecule_to_rdkit(mol)
        if rdmol is None:
            app.struct_export_status.value = "RDKit could not parse the structure."
            return
        pdb_block = Chem.MolToPDBBlock(rdmol)
        dest = (app._last_result_dir / fname) if app._last_result_dir else Path(fname)
        dest.write_text(pdb_block, encoding="utf-8")
        app.struct_export_status.value = f"Saved: {dest}"
    except Exception as exc:
        app.struct_export_status.value = f"Error: {exc}"


def on_export_reorg_geometries(app: Any, btn: Any) -> None:
    """Export every retained reorg-energy geometry as its own XYZ file
    (M-EXPORT2 EXP2.1).

    ``app._reorg_geometries`` (built by ``reorg_geometries()``) is already
    deduplicated to the DISTINCT geometries behind a run — R_neutral once,
    plus one R_ion per channel — so this writes exactly that many files, not
    one per energy. Naming follows EXP2.1's request directly:
    ``<formula>_R_neutral_<method>_<basis>.xyz`` /
    ``<formula>_R_hole_<method>_<basis>.xyz`` rather than three files all
    called ``geometry.xyz``.

    Provenance (EXP2.4): each file's comment line carries the charge,
    multiplicity, method/basis, and which of the four λ energies were
    evaluated on that geometry (``note`` — already computed by
    ``reorg_geometries()`` for the viewer, reused here rather than
    recomputed) — the whole point of a free-text XYZ comment line, and
    otherwise unrecoverable from the file six months later.
    """
    status = getattr(app, "_reorg_export_status", None)

    def _set_status(msg: str) -> None:
        if status is not None:
            status.value = msg

    geoms = getattr(app, "_reorg_geometries", None)
    if not geoms:
        _set_status("No geometries to export yet.")
        return

    from quantui.molecule import Molecule

    method = app.method_dd.value
    basis = app.basis_dd.value
    saved: list[str] = []
    try:
        for g in geoms:
            # "R_neutral — optimized neutral" -> "R_neutral"
            tag = g["label"].split(" — ")[0]
            mol = Molecule(
                atoms=list(g["atoms"]),
                coordinates=[list(c) for c in g["coordinates"]],
                charge=int(g.get("charge", 0)),
                multiplicity=int(g.get("multiplicity", 1)),
            )
            dest = export_destination(
                app,
                "reorg geometry",
                mol.get_formula(),
                tag,
                method,
                basis,
                suffix=".xyz",
            )
            comment = (
                f"{tag}  charge={mol.charge} multiplicity={mol.multiplicity}  "
                f"{method}/{basis}  {g.get('note', '')}"
            )
            full_xyz = f"{len(mol.atoms)}\n{comment}\n{mol.to_xyz_string()}\n"
            dest.write_text(full_xyz, encoding="utf-8")
            saved.append(dest.name)
    except Exception as exc:
        _set_status(f"Error: {exc}")
        return
    _set_status(f"Saved {len(saved)} file(s): " + ", ".join(saved))


def export_molecule_and_label(app: Any) -> tuple[Any, str, str]:
    """Return (molecule, method, basis) for structure export.

    For geometry optimization results, returns the final optimized geometry.
    Falls back to the currently loaded molecule for all other calculation types.
    """
    from quantui.optimizer import OptimizationResult

    result = app._last_result
    if isinstance(result, OptimizationResult):
        mol = result.molecule
    else:
        assert app._molecule is not None
        mol = app._molecule
    method = (
        getattr(result, "method", app.method_dd.value)
        if result is not None
        else app.method_dd.value
    )
    basis = (
        getattr(result, "basis", app.basis_dd.value)
        if result is not None
        else app.basis_dd.value
    )
    return mol, method, basis


# Data URIs from 3Dmol's pngURI(). Anchored and length-bounded: this value
# arrives from the browser, and while it is the user's own page rather than an
# untrusted party, decoding whatever lands in a widget traitlet without
# checking its shape is not a habit worth having.
logger = logging.getLogger(__name__)

_PNG_URI_PREFIX = "data:image/png;base64,"
_MAX_PNG_BYTES = 64 * 1024 * 1024


def _requested_dpi(app: Any) -> int:
    try:
        return int(getattr(app._iso_png_dpi, "value", 300))
    except Exception:  # noqa: BLE001 — a missing widget must not stop a save
        return 300


def _with_dpi(
    raw: bytes, dpi: Optional[int], *, metadata: Optional[dict[str, str]] = None
) -> bytes:
    """Stamp *dpi* into the PNG's pHYs chunk, and optionally provenance
    (M-EXPORT2 EXP2.4) into ``tEXt`` chunks — one re-encode for both, since
    Pillow does both in the same ``save()`` call. *dpi* of ``None`` skips the
    pHYs stamp and writes metadata only — for exporters that intentionally
    don't offer a DPI setting (e.g. the reorg-geometry PNG, which deliberately
    doesn't inherit the isosurface panel's DPI control).

    ⚠️ This sets the PRINT size, not the pixel count. The capture is whatever
    the canvas holds, and re-encoding cannot invent detail — at 300 dpi a
    760 px image simply declares itself 2.5 inches wide. That is what makes a
    figure land at the right physical size in Word or LaTeX instead of being
    scaled by hand, which is the actual complaint DPI settings answer.

    *metadata*, if given, becomes one ``tEXt`` chunk per key — method, basis,
    grid resolution, whatever the caller has. A PNG has no comment-line
    equivalent to an XYZ or cube file, so this is the export's only chance to
    carry that context; without it a figure someone emailed you a year later
    is orphaned from what produced it, same argument as ORBX.4 for cubes.
    Read back with ``PIL.Image.open(path).text``.

    Pillow is already a dependency (via matplotlib). If anything goes wrong the
    original bytes are returned: a PNG without the metadata is a mild loss, a
    failed export is not.
    """
    try:
        import io

        from PIL import Image
        from PIL.PngImagePlugin import PngInfo

        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            buf = io.BytesIO()
            pnginfo = None
            if metadata:
                pnginfo = PngInfo()
                for key, value in metadata.items():
                    if value:
                        pnginfo.add_text(key, str(value))
            save_kwargs: dict[str, Any] = {"format": "PNG", "pnginfo": pnginfo}
            if dpi is not None:
                save_kwargs["dpi"] = (dpi, dpi)
            # RGBA is preserved, so a transparent capture stays transparent.
            im.save(buf, **save_kwargs)
            return buf.getvalue()
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not finalize PNG (dpi=%s): %s", dpi, exc)
        return raw


def on_orb_png_captured(app: Any, change: dict) -> None:
    """Write a PNG captured from the live isosurface viewer (ORBX.1).

    Fires when the viewer's Save-PNG button writes a data URI into the hidden
    inbox Textarea. The image is whatever the user is actually looking at,
    camera included — which is the entire reason capture happens client-side
    rather than by re-rendering server-side.

    The inbox is cleared afterwards so saving the same view twice re-triggers
    the traitlet (an unchanged value would not fire ``observe``).
    """
    import base64
    import binascii

    uri = (change or {}).get("new") or ""
    if not uri:
        return

    def _fail(msg: str) -> None:
        app._iso_export_status.value = f'<span style="color:#b22">{msg}</span>'
        # Clear even on failure, or a retry of the identical capture is silent.
        _clear_inbox(app)

    def _clear_inbox(a: Any) -> None:
        box = getattr(a, "_orb_png_inbox", None)
        if box is not None and box.value:
            box.value = ""

    if not uri.startswith(_PNG_URI_PREFIX):
        logger.warning("PNG capture: unexpected data URI prefix")
        _fail("Capture failed (unexpected image format).")
        return
    if len(uri) > _MAX_PNG_BYTES:
        _fail("Capture failed (image too large).")
        return

    result_dir = getattr(app, "_last_result_dir", None)
    if result_dir is None or not isinstance(result_dir, Path):
        _fail("No result folder available.")
        return

    try:
        raw = base64.b64decode(uri[len(_PNG_URI_PREFIX) :], validate=True)
    except (binascii.Error, ValueError) as exc:
        logger.warning("PNG capture: could not decode payload: %s", exc)
        _fail("Capture failed (corrupt image data).")
        return

    # Filename: the user's, falling back to the orbital label. Sanitised
    # either way — this builds a filesystem path, and a name typed into a text
    # box is exactly where a stray "../" or "/" arrives.
    # str() rather than trusting the attribute: this runs against real widgets
    # in the app but also against partially-built ones, and a non-string here
    # would turn a successful capture into a traceback on the .strip() below.
    typed = getattr(getattr(app, "_iso_png_name", None), "value", "")
    typed = typed.strip() if isinstance(typed, str) else ""
    label = typed or getattr(app, "_last_cube_orbital", None) or "orbital"
    label = str(label)
    if label.lower().endswith(".png"):
        label = label[:-4]
    safe = "".join(c if c.isalnum() or c in "-_+ " else "_" for c in str(label))
    safe = safe.strip() or "orbital"
    dest = Path(result_dir) / f"{safe}.png"

    # Provenance (M-EXPORT2 EXP2.4 / M-ORBEXPORT ORBX.4): best-effort from
    # the live UI state, same caveat as the cube path — not re-verified
    # against what actually produced the captured image (e.g. after a
    # History replay). The resolution key is read directly from the dropdown
    # here rather than reverse-mapped from a grid size, since this call site
    # has the preset name itself.
    metadata = {
        "Software": "QuantUI",
        "Orbital": label,
        "Method": str(getattr(getattr(app, "method_dd", None), "value", "") or ""),
        "Basis": str(getattr(getattr(app, "basis_dd", None), "value", "") or ""),
        "Isosurface resolution": str(
            getattr(getattr(app, "_iso_resolution_dd", None), "value", "") or ""
        ),
    }
    raw = _with_dpi(raw, _requested_dpi(app), metadata=metadata)
    try:
        dest.write_bytes(raw)
    except OSError as exc:
        logger.warning("PNG capture: could not write %s: %s", dest, exc)
        _fail("Could not write the image (see log).")
        return

    logger.info("Saved orbital PNG: %s (%d bytes)", dest, len(raw))
    app._iso_export_status.value = f'<span style="color:#2a7">Saved: {dest.name}</span>'
    _clear_inbox(app)


def on_reorg_png_captured(app: Any, change: dict) -> None:
    """Write a PNG captured from the live reorg-geometry viewer (M-EXPORT2 EXP2.2).

    Same capture/decode/save shape as ``on_orb_png_captured``, fed by its own
    inbox (``_reorg_png_inbox`` / ``_REORG_PNG_INBOX_CLASS``) so a capture from
    this viewer is never mistaken for an isosurface capture. Filename/location
    goes through ``export_destination`` (EXP2.3) since this is a new exporter,
    not a retrofit of existing behaviour.

    Deliberately skips the isosurface panel's DPI-stamping/custom-name extras
    (``_iso_png_dpi`` / ``_iso_png_name``) — those are isosurface-panel
    controls, and wiring them in here would make an unrelated panel's PNG
    export depend on a setting the user set for a different viewer.
    """
    import base64
    import binascii

    uri = (change or {}).get("new") or ""
    if not uri:
        return

    status = getattr(app, "_reorg_export_status", None)

    def _fail(msg: str) -> None:
        if status is not None:
            status.value = f'<span style="color:#b22">{msg}</span>'
        _clear_inbox(app)

    def _clear_inbox(a: Any) -> None:
        box = getattr(a, "_reorg_png_inbox", None)
        if box is not None and box.value:
            box.value = ""

    if not uri.startswith(_PNG_URI_PREFIX):
        logger.warning("reorg PNG capture: unexpected data URI prefix")
        _fail("Capture failed (unexpected image format).")
        return
    if len(uri) > _MAX_PNG_BYTES:
        _fail("Capture failed (image too large).")
        return

    try:
        raw = base64.b64decode(uri[len(_PNG_URI_PREFIX) :], validate=True)
    except (binascii.Error, ValueError) as exc:
        logger.warning("reorg PNG capture: could not decode payload: %s", exc)
        _fail("Capture failed (corrupt image data).")
        return

    mol = getattr(app, "_molecule", None)
    formula = mol.get_formula() if mol is not None else "molecule"
    method = str(getattr(getattr(app, "method_dd", None), "value", "") or "")
    basis = str(getattr(getattr(app, "basis_dd", None), "value", "") or "")

    try:
        dest = export_destination(
            app, "reorg geometry PNG", formula, "geometry", method, basis, suffix=".png"
        )
    except ValueError as exc:
        _fail(str(exc))
        return

    # Provenance (M-EXPORT2 EXP2.4) — metadata only, no DPI stamp (see
    # _with_dpi's docstring for why this exporter skips DPI intentionally).
    raw = _with_dpi(
        raw,
        None,
        metadata={"Software": "QuantUI", "Method": method, "Basis": basis},
    )
    try:
        dest.write_bytes(raw)
    except OSError as exc:
        logger.warning("reorg PNG capture: could not write %s: %s", dest, exc)
        _fail("Could not write the image (see log).")
        return

    logger.info("Saved reorg geometry PNG: %s (%d bytes)", dest, len(raw))
    if status is not None:
        status.value = f'<span style="color:#2a7">Saved: {dest.name}</span>'
    _clear_inbox(app)


def on_vib_png_captured(app: Any, change: dict) -> None:
    """Write a PNG captured from the live vibrational single-viewer
    (M-EXPORT2 EXP2.2).

    Same capture/decode/save shape as ``on_reorg_png_captured``, fed by its
    own inbox (``_vib_png_inbox`` / ``_VIB_PNG_INBOX_CLASS``). Only the
    single-persistent-viewer (py3Dmol) path renders the Save-PNG button — see
    ``build_vib_viewer_html`` — so this never fires from the legacy per-mode
    plotlymol3d fallback, which has no equivalent capture bridge.
    """
    import base64
    import binascii

    uri = (change or {}).get("new") or ""
    if not uri:
        return

    status = getattr(app, "_vib_png_status", None)

    def _fail(msg: str) -> None:
        if status is not None:
            status.value = f'<span style="color:#b22">{msg}</span>'
        _clear_inbox(app)

    def _clear_inbox(a: Any) -> None:
        box = getattr(a, "_vib_png_inbox", None)
        if box is not None and box.value:
            box.value = ""

    if not uri.startswith(_PNG_URI_PREFIX):
        logger.warning("vib PNG capture: unexpected data URI prefix")
        _fail("Capture failed (unexpected image format).")
        return
    if len(uri) > _MAX_PNG_BYTES:
        _fail("Capture failed (image too large).")
        return

    try:
        raw = base64.b64decode(uri[len(_PNG_URI_PREFIX) :], validate=True)
    except (binascii.Error, ValueError) as exc:
        logger.warning("vib PNG capture: could not decode payload: %s", exc)
        _fail("Capture failed (corrupt image data).")
        return

    mol = getattr(app, "_last_vib_molecule", None)
    formula = mol.get_formula() if mol is not None else "molecule"

    mode_label = "mode"
    freq_cm1: str = ""
    raw_mode = getattr(getattr(app, "vib_mode_dd", None), "value", None)
    try:
        if raw_mode is None:
            raise TypeError("no mode selected")
        mode_number = int(raw_mode)
        mode_label = f"mode{mode_number}"
        freq_result = getattr(app, "_last_vib_freq_result", None)
        freqs = getattr(freq_result, "frequencies_cm1", None) if freq_result else None
        if freqs and 0 < mode_number <= len(freqs):
            freq_cm1 = f"{freqs[mode_number - 1]:.1f}"
    except (TypeError, ValueError):
        pass  # no mode selected yet — fall back to the generic "mode" label

    method = str(getattr(getattr(app, "method_dd", None), "value", "") or "")
    basis = str(getattr(getattr(app, "basis_dd", None), "value", "") or "")

    try:
        dest = export_destination(
            app, "vibrational mode PNG", formula, mode_label, suffix=".png"
        )
    except ValueError as exc:
        _fail(str(exc))
        return

    # Provenance (M-EXPORT2 EXP2.4) — same argument as the reorg/trajectory
    # exporters: cheap now, unrecoverable once the file has left the machine.
    metadata = {"Software": "QuantUI", "Method": method, "Basis": basis}
    if freq_cm1:
        metadata["Frequency (cm-1)"] = freq_cm1
    raw = _with_dpi(raw, None, metadata=metadata)
    try:
        dest.write_bytes(raw)
    except OSError as exc:
        logger.warning("vib PNG capture: could not write %s: %s", dest, exc)
        _fail("Could not write the image (see log).")
        return

    logger.info("Saved vibrational mode PNG: %s (%d bytes)", dest, len(raw))
    if status is not None:
        status.value = f'<span style="color:#2a7">Saved: {dest.name}</span>'
    _clear_inbox(app)


def _on_mol_png_captured(
    app: Any,
    change: dict,
    *,
    inbox_attr: str,
    status_attr: str,
    molecule: Any,
    slot_label: str,
) -> None:
    """Shared implementation for the molecule (top) viewer's three Save-PNG
    buttons (M-EXPORT2 EXP2.2) — Calculate-tab preview, Results-tab viewer,
    Analysis-tab viewer. These are the same underlying viewer
    (``visualization_py3dmol.render_molecule_html``) rendered into three
    independent output slots, not three different viewers, so one shared
    implementation parameterized by which slot fired beats three near-copies
    of ``on_reorg_png_captured``'s shape.

    Each slot has its own persistent inbox/status attribute (never shared —
    see the class-name comment on ``_MOL_*_PNG_INBOX_CLASS`` in
    ``app_builders.py`` for why) and its own ``slot_label`` so a capture from
    two slots showing the same molecule doesn't overwrite the same filename.
    ``molecule`` is resolved by the caller at fire time (``app._molecule``
    for Calculate/Results, ``app._analysis_displayed_molecule`` for
    Analysis) since the Analysis-tab viewer can show a different molecule
    than the one currently loaded on the Calculate tab (e.g. after a History
    replay).
    """
    import base64
    import binascii

    uri = (change or {}).get("new") or ""
    if not uri:
        return

    status = getattr(app, status_attr, None)

    def _fail(msg: str) -> None:
        if status is not None:
            status.value = f'<span style="color:#b22">{msg}</span>'
        _clear_inbox(app)

    def _clear_inbox(a: Any) -> None:
        box = getattr(a, inbox_attr, None)
        if box is not None and box.value:
            box.value = ""

    if not uri.startswith(_PNG_URI_PREFIX):
        logger.warning(
            "molecule PNG capture (%s): unexpected data URI prefix", slot_label
        )
        _fail("Capture failed (unexpected image format).")
        return
    if len(uri) > _MAX_PNG_BYTES:
        _fail("Capture failed (image too large).")
        return

    try:
        raw = base64.b64decode(uri[len(_PNG_URI_PREFIX) :], validate=True)
    except (binascii.Error, ValueError) as exc:
        logger.warning(
            "molecule PNG capture (%s): could not decode payload: %s", slot_label, exc
        )
        _fail("Capture failed (corrupt image data).")
        return

    formula = molecule.get_formula() if molecule is not None else "molecule"
    method = str(getattr(getattr(app, "method_dd", None), "value", "") or "")
    basis = str(getattr(getattr(app, "basis_dd", None), "value", "") or "")

    try:
        dest = export_destination(
            app,
            "molecule PNG",
            formula,
            slot_label,
            suffix=".png",
            general_if_no_result=True,
            timestamp=True,
        )
    except ValueError as exc:
        _fail(str(exc))
        return

    # Provenance (M-EXPORT2 EXP2.4) — same argument as every other exporter
    # in this module: cheap now, unrecoverable once the file has left the
    # machine.
    raw = _with_dpi(
        raw,
        None,
        metadata={"Software": "QuantUI", "Method": method, "Basis": basis},
    )
    try:
        dest.write_bytes(raw)
    except OSError as exc:
        logger.warning(
            "molecule PNG capture (%s): could not write %s: %s", slot_label, dest, exc
        )
        _fail("Could not write the image (see log).")
        return

    logger.info("Saved molecule PNG (%s): %s (%d bytes)", slot_label, dest, len(raw))
    if status is not None:
        status.value = f'<span style="color:#2a7">Saved: {dest.name}</span>'
    _clear_inbox(app)


def on_mol_calc_png_captured(app: Any, change: dict) -> None:
    """Save-PNG button on the Calculate-tab molecule preview."""
    _on_mol_png_captured(
        app,
        change,
        inbox_attr="_mol_calc_png_inbox",
        status_attr="_mol_calc_png_status",
        molecule=getattr(app, "_molecule", None),
        slot_label="calc",
    )


def on_mol_results_png_captured(app: Any, change: dict) -> None:
    """Save-PNG button on the Results-tab molecule viewer."""
    _on_mol_png_captured(
        app,
        change,
        inbox_attr="_mol_results_png_inbox",
        status_attr="_mol_results_png_status",
        molecule=getattr(app, "_molecule", None),
        slot_label="results",
    )


def on_mol_analysis_png_captured(app: Any, change: dict) -> None:
    """Save-PNG button on the Analysis-tab molecule viewer.

    Uses ``_analysis_displayed_molecule``, not ``_molecule`` — the Analysis
    tab can show a molecule from a History replay that differs from whatever
    is currently loaded on the Calculate tab.
    """
    _on_mol_png_captured(
        app,
        change,
        inbox_attr="_mol_analysis_png_inbox",
        status_attr="_mol_analysis_png_status",
        molecule=getattr(app, "_analysis_displayed_molecule", None),
        slot_label="analysis",
    )


def on_traj_png_captured(
    app: Any, change: dict, *, formula: str = "", status: Any = None
) -> None:
    """Write a PNG captured from the live trajectory viewer (M-EXPORT2 EXP2.2).

    Same capture/decode/save shape as ``on_reorg_png_captured``, but the
    trajectory panel's widgets (unlike the isosurface/reorg accordions) are
    rebuilt fresh on every render rather than constructed once in
    ``app_builders``, so there is no persistent ``app._traj_png_inbox`` /
    ``app._traj_export_status`` to look up: the inbox to clear and the status
    label to update are passed in directly by the caller
    (``app_visualization.show_opt_trajectory``), which is the one place that
    still holds a reference to this render's widgets. Clearing reads the
    firing widget straight off ``change["owner"]`` for the same reason.
    """
    import base64
    import binascii

    uri = (change or {}).get("new") or ""
    if not uri:
        return

    inbox = (change or {}).get("owner")

    def _fail(msg: str) -> None:
        if status is not None:
            status.value = f'<span style="color:#b22">{msg}</span>'
        _clear_inbox()

    def _clear_inbox() -> None:
        if inbox is None:
            return
        if getattr(inbox, "value", ""):
            inbox.value = ""

    if not uri.startswith(_PNG_URI_PREFIX):
        logger.warning("trajectory PNG capture: unexpected data URI prefix")
        _fail("Capture failed (unexpected image format).")
        return
    if len(uri) > _MAX_PNG_BYTES:
        _fail("Capture failed (image too large).")
        return

    try:
        raw = base64.b64decode(uri[len(_PNG_URI_PREFIX) :], validate=True)
    except (binascii.Error, ValueError) as exc:
        logger.warning("trajectory PNG capture: could not decode payload: %s", exc)
        _fail("Capture failed (corrupt image data).")
        return

    formula = formula or "molecule"
    method = str(getattr(getattr(app, "method_dd", None), "value", "") or "")
    basis = str(getattr(getattr(app, "basis_dd", None), "value", "") or "")

    try:
        dest = export_destination(
            app, "trajectory PNG", formula, "trajectory", suffix=".png"
        )
    except ValueError as exc:
        _fail(str(exc))
        return

    # Provenance (M-EXPORT2 EXP2.4) — same argument as the reorg exporter:
    # cheap now, unrecoverable once the file has left the machine.
    raw = _with_dpi(
        raw,
        None,
        metadata={"Software": "QuantUI", "Method": method, "Basis": basis},
    )
    try:
        dest.write_bytes(raw)
    except OSError as exc:
        logger.warning("trajectory PNG capture: could not write %s: %s", dest, exc)
        _fail("Could not write the image (see log).")
        return

    logger.info("Saved trajectory PNG: %s (%d bytes)", dest, len(raw))
    if status is not None:
        status.value = f'<span style="color:#2a7">Saved: {dest.name}</span>'
    _clear_inbox()


def on_iso_export_cube(app: Any, btn: Any) -> None:
    """Copy the last-generated cube file to the result folder.

    Reads ``app._last_cube_path`` (set by the isosurface render path in
    ``app_visualization.py``) and copies it to
    ``<result_dir>/<orbital_label>.cube`` so the user can hand a
    friendly-named cube to Avogadro / VMD / Multiwfn without scrolling
    through ``isosurfaces/<formula>_<orb>_<timestamp>.cube``.
    """
    from quantui.results_storage import export_cube

    src = getattr(app, "_last_cube_path", None)
    label = getattr(app, "_last_cube_orbital", None) or "orbital"
    result_dir = getattr(app, "_last_result_dir", None)
    if src is None or not isinstance(src, Path) or not src.exists():
        app._iso_export_status.value = (
            '<span style="color:#b22">Generate an isosurface first.</span>'
        )
        return
    if result_dir is None or not isinstance(result_dir, Path):
        app._iso_export_status.value = (
            '<span style="color:#b22">No result folder available.</span>'
        )
        return
    dest = export_cube(src, result_dir, orbital_label=label)
    if dest is None:
        app._iso_export_status.value = (
            '<span style="color:#b22">Cube export failed (see log).</span>'
        )
        return
    app._iso_export_status.value = f'<span style="color:#2a7">Saved: {dest.name}</span>'


def on_export_bundle(app: Any, btn: Any) -> None:
    """Zip the entire result folder for sharing."""
    from quantui.results_storage import export_result_bundle

    result_dir = getattr(app, "_last_result_dir", None)
    if result_dir is None or not isinstance(result_dir, Path):
        app._export_bundle_status.value = "Run or load a calculation first."
        return
    out_path = export_result_bundle(result_dir)
    if out_path is None:
        app._export_bundle_status.value = "Bundle export failed (see log)."
        return
    app._export_bundle_status.value = f"Saved: {out_path}"


def molecule_to_rdkit(mol: Any) -> Any:
    """Convert a Molecule to an RDKit Mol with inferred bonds (best-effort)."""
    try:
        from rdkit import Chem

        xyz_block = f"{len(mol.atoms)}\n{mol.get_formula()}\n{mol.to_xyz_string()}\n"
        rdmol = Chem.MolFromXYZBlock(xyz_block)
        if rdmol is None:
            return None
        try:
            from rdkit.Chem import rdDetermineBonds

            rdDetermineBonds.DetermineBonds(rdmol, charge=mol.charge)
        except Exception:
            pass
        return rdmol
    except Exception:
        return None
