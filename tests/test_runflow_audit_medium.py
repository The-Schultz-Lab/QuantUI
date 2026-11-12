"""Run-flow fixes from the 2026-11 audit (Medium findings).

- APP-5: a failed save is reported, not swallowed; a failed extra file
  (thumbnail, Molden, ...) does not unsave the result.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from quantui.app import QuantUIApp
from quantui.molecule import Molecule

pyscf = pytest.importorskip("pyscf")


def _h2() -> Molecule:
    return Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])


@pytest.fixture
def app():
    a = QuantUIApp()
    a._set_molecule(_h2())
    a.method_dd.value = "RHF"
    a.basis_dd.value = "STO-3G"
    a.calc_type_dd.value = "Single Point"
    return a


def _run(app: QuantUIApp) -> None:
    app._begin_run_lock()
    app._run_in_thread()


def _result_html(app: QuantUIApp) -> str:
    return "\n".join(
        str(o.get("data", {}).get("text/html", "")) for o in app.result_output.outputs
    )


# ── APP-5 ─────────────────────────────────────────────────────────────────


def test_failed_save_is_reported_and_forgets_the_old_folder(app, tmp_path):
    app._last_result_dir = tmp_path  # a previous run's folder
    with patch(
        "quantui.save_result", side_effect=OSError(28, "No space left on device")
    ):
        _run(app)
    assert "NOT saved" in app.run_status.value
    assert "No space left" in app.run_status.value
    assert not app.run_status.value.startswith("Done")
    assert app._last_result_dir is None
    assert app._export_bundle_btn.disabled
    assert "Result NOT saved" in _result_html(app)
    # The calculation itself still finished and is shown.
    assert app._last_result is not None


def test_failed_extra_file_does_not_unsave_the_result(app, tmp_path):
    saved = tmp_path / "run"
    saved.mkdir()
    with patch("quantui.save_result", return_value=saved), patch(
        "quantui.load_result", return_value={"timestamp": "t"}
    ), patch(
        "quantui.results_storage.save_thumbnail", side_effect=OSError("boom")
    ), patch(
        "quantui.results_storage.save_orbitals"
    ) as orbitals, patch(
        "quantui.results_storage.save_molden"
    ) as molden:
        _run(app)
    assert app.run_status.value.startswith("Done"), app.run_status.value
    assert app._last_result_dir == saved
    # The extras after the failing thumbnail still ran.
    orbitals.assert_called_once()
    molden.assert_called_once()
    assert "NOT saved" not in _result_html(app)
    assert isinstance(app._last_result_dir, Path)
