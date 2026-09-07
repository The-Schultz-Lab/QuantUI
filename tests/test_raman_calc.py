"""Tests for quantui.raman_calc — Raman activity invariants and integration."""

from __future__ import annotations

import numpy as np
import pytest

from quantui.raman_calc import (
    _raman_invariants,
    _try_gpu_raman_activities,
    compute_raman_activities,
    raman_enabled,
)


class TestRamanInvariants:
    def test_isotropic_mode_gives_45_alpha_bar_squared(self):
        # ∂α/∂Q proportional to identity → γ² = 0
        da = np.eye(3) * 3.0
        assert _raman_invariants(da) == pytest.approx(45.0 * 3.0**2)

    def test_traceless_anisotropic_contributes_via_gamma(self):
        da = np.diag([1.0, -1.0, 0.0])
        alpha_bar = 0.0
        gamma2 = 0.5 * ((1 - (-1)) ** 2 + ((-1) - 0) ** 2 + (0 - 1) ** 2)
        expected = 45 * alpha_bar**2 + 7 * gamma2
        assert _raman_invariants(da) == pytest.approx(expected)

    def test_non_negative(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            da = rng.normal(size=(3, 3))
            da = 0.5 * (da + da.T)
            assert _raman_invariants(da) >= 0.0


class TestRamanUnitsRegression:
    """AUDIT F02 — CPU Raman activities were ~45.54x too large because the
    polarizability numerator (a0^3) was never converted to Angstrom^3, only
    the displacement denominator (Bohr -> Angstrom) was. This reproduces the
    audit's own method: an independent central difference of the real SCF
    polarizability along the complete (mass-normalized) normal mode, using
    none of raman_calc's atom-by-atom Jacobian/unit-conversion code.
    """

    @pytest.mark.slow
    def test_h2_raman_activity_matches_independent_normal_mode_fd(self):
        pyscf = pytest.importorskip("pyscf")
        pytest.importorskip("pyscf.prop.polarizability.rhf")
        import numpy as _np
        from pyscf import gto, scf
        from pyscf.prop.polarizability import rhf as pol_mod

        from quantui.config import BOHR_TO_ANGSTROM as _BOHR_TO_ANG
        from quantui.freq_calc import run_freq_calc
        from quantui.molecule import Molecule

        bond_length = 0.74  # Angstrom
        h2 = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, bond_length]])
        result = run_freq_calc(h2, method="RHF", basis="STO-3G")
        assert result.raman_activities, "H2/RHF/STO-3G should produce a Raman activity"
        assert result.displacements, "normal-mode displacements required for FD check"

        nm_flat = _np.asarray(result.displacements[0], dtype=float).reshape(-1)

        def _alpha_au(coords_bohr: _np.ndarray) -> _np.ndarray:
            mol = gto.M(
                atom=[
                    ("H", tuple(coords_bohr[0])),
                    ("H", tuple(coords_bohr[1])),
                ],
                basis="STO-3G",
                unit="Bohr",
                verbose=0,
            )
            mf = scf.RHF(mol)
            mf.verbose = 0
            mf.kernel()
            return _np.asarray(
                pol_mod.polarizability(pol_mod.Polarizability(mf)), dtype=float
            )

        coords0_bohr = _np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, bond_length / _BOHR_TO_ANG]]
        )
        # Scale the step so the largest per-atom Cartesian displacement is a
        # modest 0.01 Bohr, regardless of how PySCF normalizes the mode.
        eps = 0.01 / max(abs(nm_flat.max()), abs(nm_flat.min()), 1e-12)
        disp_bohr = (eps * nm_flat).reshape(-1, 3)

        alpha_plus = _alpha_au(coords0_bohr + disp_bohr)
        alpha_minus = _alpha_au(coords0_bohr - disp_bohr)
        # d(alpha[a0^3]) / d(eps) == sum_k nm_k * d(alpha[a0^3])/d(x_k[Bohr])
        # by the chain rule — exactly the per-atom Jacobian raman_calc.py
        # projects onto the same nm vector, just computed by directly
        # perturbing along the mode instead of atom-by-atom.
        dalpha_dq_au = (alpha_plus - alpha_minus) / (2.0 * eps)
        dalpha_dq_ang = dalpha_dq_au * (_BOHR_TO_ANG**2)

        expected_activity = _raman_invariants(dalpha_dq_ang)
        assert result.raman_activities[0] == pytest.approx(
            expected_activity, rel=0.05
        )
        # The pre-fix bug deflated this by ~45.54x — well outside any
        # plausible finite-difference discrepancy.
        assert result.raman_activities[0] > 0.2 * expected_activity


class TestRamanEnabled:
    def test_default_enabled(self, monkeypatch):
        monkeypatch.delenv("QUANTUI_RAMAN", raising=False)
        assert raman_enabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "off"])
    def test_opt_out_values(self, monkeypatch, val):
        monkeypatch.setenv("QUANTUI_RAMAN", val)
        assert raman_enabled() is False


class TestGpuRamanDispatch:
    def test_gpu_path_returns_when_available(self, monkeypatch):
        calls: list[str] = []

        def _status(msg: str) -> None:
            calls.append(msg)

        monkeypatch.setattr(
            "quantui.raman_calc.is_gpu_available",
            lambda: (True, "TestGPU"),
        )

        def _fake_eval(mf_gpu, hessian=None):
            return ([100.0, 200.0], [10.0, 20.0], [0.1, 0.2])

        import sys
        import types

        fake_mod = types.ModuleType("gpu4pyscf.properties.raman")
        fake_mod.eval_raman_intensity = _fake_eval
        fake_pkg = types.ModuleType("gpu4pyscf.properties")
        fake_pkg.raman = fake_mod
        fake_root = types.ModuleType("gpu4pyscf")
        fake_root.properties = fake_pkg
        monkeypatch.setitem(sys.modules, "gpu4pyscf", fake_root)
        monkeypatch.setitem(sys.modules, "gpu4pyscf.properties", fake_pkg)
        monkeypatch.setitem(sys.modules, "gpu4pyscf.properties.raman", fake_mod)

        sentinel_mf = type("MF", (), {"to_gpu": lambda self: self})()
        out = _try_gpu_raman_activities(
            mf=sentinel_mf,
            hessian=object(),
            frequencies_cm1=[100.0, 200.0],
            dm0_is_unrestricted=False,
            status=_status,
        )
        assert out == [10.0, 20.0]
        assert any("gpu4pyscf" in c for c in calls)

    def test_gpu_skipped_for_unrestricted(self, monkeypatch):
        monkeypatch.setattr(
            "quantui.raman_calc.is_gpu_available",
            lambda: (True, "TestGPU"),
        )
        called = {"gpu": False}

        def _boom(*_a, **_k):
            called["gpu"] = True
            raise AssertionError("should not call gpu raman")

        import sys
        import types

        fake_mod = types.ModuleType("gpu4pyscf.properties.raman")
        fake_mod.eval_raman_intensity = _boom
        fake_pkg = types.ModuleType("gpu4pyscf.properties")
        fake_pkg.raman = fake_mod
        fake_root = types.ModuleType("gpu4pyscf")
        fake_root.properties = fake_pkg
        monkeypatch.setitem(sys.modules, "gpu4pyscf", fake_root)
        monkeypatch.setitem(sys.modules, "gpu4pyscf.properties", fake_pkg)
        monkeypatch.setitem(sys.modules, "gpu4pyscf.properties.raman", fake_mod)

        assert (
            _try_gpu_raman_activities(
                mf=object(),
                hessian=object(),
                frequencies_cm1=[1.0],
                dm0_is_unrestricted=True,
                status=lambda _m: None,
            )
            is None
        )
        assert not called["gpu"]

    def test_compute_prefers_gpu_over_cpu(self, monkeypatch):
        monkeypatch.setenv("QUANTUI_RAMAN", "1")
        monkeypatch.setattr(
            "quantui.raman_calc._try_gpu_raman_activities",
            lambda **_kw: [1.0, 2.0, 3.0],
        )

        def _cpu_should_not_run(**_kw):
            raise AssertionError("CPU path should not run when GPU succeeds")

        monkeypatch.setattr(
            "quantui.raman_calc._cpu_raman_activities_fd",
            _cpu_should_not_run,
        )

        out = compute_raman_activities(
            mf=object(),
            mol=object(),
            scf=object(),
            dft=object(),
            displacements=[[[[0.0]]]],
            frequencies_cm1=[1.0, 2.0, 3.0],
            dm0=object(),
            dm0_is_unrestricted=False,
            density_fit_used=False,
            stream=__import__("io").StringIO(),
            status=lambda _m: None,
            hessian=object(),
        )
        assert out == [1.0, 2.0, 3.0]

    def test_compute_falls_back_when_gpu_unavailable(self, monkeypatch):
        monkeypatch.setenv("QUANTUI_RAMAN", "1")
        monkeypatch.setattr(
            "quantui.raman_calc._try_gpu_raman_activities",
            lambda **_kw: None,
        )
        monkeypatch.setattr(
            "quantui.raman_calc._cpu_raman_activities_fd",
            lambda **_kw: [4.0, 5.0],
        )

        out = compute_raman_activities(
            mf=object(),
            mol=object(),
            scf=object(),
            dft=object(),
            displacements=[[[[0.0]]]],
            frequencies_cm1=[1.0, 2.0],
            dm0=object(),
            dm0_is_unrestricted=False,
            density_fit_used=False,
            stream=__import__("io").StringIO(),
            status=lambda _m: None,
            hessian=object(),
        )
        assert out == [4.0, 5.0]


# ---------------------------------------------------------------------------
# Integration — PySCF required
# ---------------------------------------------------------------------------

try:
    import pyscf  # noqa: F401

    _HAS_PYSCF = True
except ImportError:
    _HAS_PYSCF = False

pyscf_only = pytest.mark.skipif(not _HAS_PYSCF, reason="PySCF not installed")


def _water():
    from quantui.molecule import Molecule

    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.0, 0.96, 0.0]],
        charge=0,
        multiplicity=1,
    )


class TestRamanActivitiesIntegration:
    """H₂O / RHF / STO-3G — Raman should ship alongside IR on Frequency runs."""

    @pyscf_only
    @pytest.mark.slow
    def test_raman_activities_non_empty(self, monkeypatch):
        monkeypatch.setenv("QUANTUI_RAMAN", "1")
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="RHF", basis="STO-3G")
        assert (
            result.raman_activities
        ), "raman_activities should be non-empty for H₂O/RHF"

    @pyscf_only
    @pytest.mark.slow
    def test_raman_length_matches_frequencies(self, monkeypatch):
        monkeypatch.setenv("QUANTUI_RAMAN", "1")
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="RHF", basis="STO-3G")
        assert len(result.raman_activities) == len(result.frequencies_cm1)

    @pyscf_only
    @pytest.mark.slow
    def test_raman_opt_out(self, monkeypatch):
        monkeypatch.setenv("QUANTUI_RAMAN", "0")
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="RHF", basis="STO-3G")
        assert result.raman_activities == []

    @pyscf_only
    @pytest.mark.slow
    def test_h2o_bend_weaker_ir_than_raman(self, monkeypatch):
        """Teaching check: lowest H₂O mode is weak IR, strong Raman at STO-3G."""
        monkeypatch.setenv("QUANTUI_RAMAN", "1")
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="RHF", basis="STO-3G")
        pairs = [
            (f, ir, ram)
            for f, ir, ram in zip(
                result.frequencies_cm1,
                result.ir_intensities,
                result.raman_activities,
            )
            if f > 0
        ]
        assert pairs
        lowest = min(pairs, key=lambda t: t[0])
        _freq, ir_low, ram_low = lowest
        assert (
            ir_low < ram_low
        ), f"Expected bend-like mode IR ({ir_low:.2f}) < Raman ({ram_low:.2f})"


class TestChk4RamanDisplacementCheckpointing:
    """M-CHECKPOINT CHK.4.4 — Raman shares the exact same (atom, axis, sign)
    displacement grid as the IR loop, so it needs its own named item set
    ("raman_displacements") to avoid colliding with IR's
    ("freq_displacements") — see roadmap 34's CHK.4 section."""

    def _checkpoint(self, tmp_path, molecule):
        from quantui.checkpoint import CalcIdentity, Checkpoint

        identity = CalcIdentity.from_molecule(
            molecule, calc_type="frequency", method="RHF", basis="STO-3G"
        )
        # Matches production: backends/worker.py's _begin_worker_checkpoint
        # always calls .begin() before handing a checkpoint to a calc
        # function.
        ckpt = Checkpoint(identity, root=tmp_path / "ckpt")
        ckpt.begin()
        return ckpt

    def _count_rescue_calls(self, monkeypatch):
        import quantui.scf_robust as scf_robust_mod

        real_rescue = scf_robust_mod.run_scf_with_rescue
        calls: list = []

        def _counting(*args, **kwargs):
            calls.append(1)
            return real_rescue(*args, **kwargs)

        monkeypatch.setattr(scf_robust_mod, "run_scf_with_rescue", _counting)
        return calls

    @pyscf_only
    @pytest.mark.slow
    def test_raman_displacements_recorded_under_their_own_item_set(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("QUANTUI_RAMAN", "1")
        from quantui.freq_calc import run_freq_calc

        molecule = _water()
        ckpt = self._checkpoint(tmp_path, molecule)
        run_freq_calc(molecule, method="RHF", basis="STO-3G", checkpoint=ckpt)

        ir_ids = ckpt.completed_item_ids("freq_displacements")
        raman_ids = ckpt.completed_item_ids("raman_displacements")
        assert len(ir_ids) == 18
        assert len(raman_ids) == 18
        # Same displacement grid, but the two sets never collide — reading
        # one back must not be affected by the other's records.
        assert ir_ids == raman_ids

    @pyscf_only
    @pytest.mark.slow
    def test_fully_banked_resume_skips_all_raman_recompute(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUANTUI_RAMAN", "1")
        from quantui.freq_calc import run_freq_calc

        molecule = _water()
        ckpt = self._checkpoint(tmp_path, molecule)

        calls = self._count_rescue_calls(monkeypatch)
        baseline = run_freq_calc(
            molecule, method="RHF", basis="STO-3G", checkpoint=ckpt
        )
        # reference SCF + 18 IR displacements + 18 Raman displacements.
        assert len(calls) == 1 + 18 + 18

        calls.clear()
        ckpt.begin()  # a real resubmission calls .begin() again
        resumed = run_freq_calc(
            molecule, method="RHF", basis="STO-3G", checkpoint=ckpt, resume=True
        )
        assert len(calls) == 1  # only the reference SCF
        assert resumed.raman_activities == pytest.approx(
            baseline.raman_activities, abs=1e-6
        )
