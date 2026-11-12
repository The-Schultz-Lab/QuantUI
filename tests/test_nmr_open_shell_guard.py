"""NMR open-shell / meta-GGA guard and UHF-singlet mapping (audit SCF-3).

The pyscf-properties GIAO code only handles closed-shell references and has
no meta-GGA XC branch. Before the fix every UHF run (even a closed-shell
singlet), every open-shell DFT run and every meta-GGA functional failed after
the SCF with a cryptic error ("cannot reshape array…", an empty message,
"meta-GGA"). Now: open-shell and meta-GGA are rejected up front with a clear
``ValueError`` (and the Run button refuses them), and a UHF singlet runs as
the identical RHF reference.
"""

from __future__ import annotations

import io

import pytest

from quantui.molecule import Molecule
from quantui.nmr_calc import nmr_unsupported_reason, run_nmr_calc

try:
    import pyscf.prop.nmr  # noqa: F401

    _HAS_NMR = True
except ImportError:
    _HAS_NMR = False

nmr_only = pytest.mark.skipif(not _HAS_NMR, reason="pyscf-properties missing")


def _water(charge: int = 0, mult: int = 1) -> Molecule:
    return Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.117], [0.0, 0.757, -0.467], [0.0, -0.757, -0.467]],
        charge=charge,
        multiplicity=mult,
    )


class TestNmrUnsupportedReason:
    @pytest.mark.parametrize("method", ["RHF", "UHF", "B3LYP", "PBE0", "wB97X-D"])
    def test_closed_shell_supported(self, method):
        assert nmr_unsupported_reason(method, 1) == ""

    @pytest.mark.parametrize("method", ["UHF", "B3LYP", "RHF"])
    def test_open_shell_rejected(self, method):
        assert "closed-shell" in nmr_unsupported_reason(method, 2)

    @pytest.mark.parametrize("method", ["M06-L", "M06-2X"])
    def test_meta_gga_rejected(self, method):
        assert "meta-GGA" in nmr_unsupported_reason(method, 1)

    def test_post_hf_rejected(self):
        assert "post-HF" in nmr_unsupported_reason("MP2", 1)


class TestRunNmrCalcGuard:
    @pytest.mark.parametrize("method", ["UHF", "B3LYP"])
    def test_doublet_raises_before_scf(self, method, monkeypatch):
        # The guard must fire before any SCF work.
        import quantui.nmr_calc as nc

        monkeypatch.setattr(
            nc,
            "_run_nmr_calc_body",
            lambda **_: pytest.fail("SCF body reached for an open-shell NMR run"),
        )
        with pytest.raises(ValueError, match="closed-shell"):
            run_nmr_calc(_water(charge=1, mult=2), method=method, basis="STO-3G")

    @pytest.mark.parametrize("method", ["M06-L", "M06-2X"])
    def test_meta_gga_raises(self, method):
        with pytest.raises(ValueError, match="meta-GGA"):
            run_nmr_calc(_water(), method=method, basis="STO-3G")

    @nmr_only
    def test_uhf_singlet_runs_as_rhf(self):
        log = io.StringIO()
        uhf = run_nmr_calc(_water(), "UHF", "STO-3G", progress_stream=log)
        rhf = run_nmr_calc(_water(), "RHF", "STO-3G", progress_stream=io.StringIO())
        assert uhf.scf_variant == "RHF"
        assert uhf.method == "UHF"
        assert uhf.shielding_iso_ppm == pytest.approx(rhf.shielding_iso_ppm)
        assert uhf.reference_key == rhf.reference_key
        assert "identical to RHF" in log.getvalue()


class TestRunButtonGate:
    @pytest.fixture
    def app(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(tmp_path / "settings.json"))
        from quantui.app import QuantUIApp

        return QuantUIApp()

    def _run_text(self, app) -> str:
        return "".join(o.get("text", "") for o in app.run_output.outputs)

    def test_meta_gga_nmr_not_started(self, app):
        app._molecule = _water()
        app.calc_type_dd.value = "NMR Shielding"
        app.method_dd.value = "M06-L"
        app.basis_dd.value = "STO-3G"
        app._on_run_clicked(None)
        assert "NMR shielding was not started" in self._run_text(app)
        assert "meta-GGA" in self._run_text(app)
        assert not getattr(app, "_calc_running", False)

    def test_open_shell_nmr_not_started(self, app):
        app._molecule = _water(charge=1, mult=2)
        app.calc_type_dd.value = "NMR Shielding"
        app.method_dd.value = "B3LYP"
        app.basis_dd.value = "STO-3G"
        app.charge_si.value = 1
        app.mult_si.value = 2
        app._on_run_clicked(None)
        assert "closed-shell" in self._run_text(app)
        assert not getattr(app, "_calc_running", False)
