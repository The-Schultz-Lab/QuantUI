"""PyFock Phase-1 adapter unit tests and opt-in native integration check."""

from __future__ import annotations

import io
import os
import sys
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pytest

from quantui.engines import EngineRequest, PyfockEngine, UnsupportedCapabilityError


def _request(**overrides):
    data = {
        "request_id": "water-pbe",
        "calc_type": "single_point",
        "method": "PBE",
        "basis": "def2-SVP",
        "charge": 0,
        "multiplicity": 1,
        "molecule": {
            "atoms": ["O", "H", "H"],
            "coordinates": [
                [0.000000, 0.000000, 0.117790],
                [0.000000, 0.755453, -0.471161],
                [0.000000, -0.755453, -0.471161],
            ],
        },
    }
    data.update(overrides)
    return EngineRequest(**data)


class _FakeMol:
    def __init__(self, *, atoms, charge):
        self.atoms = atoms
        self.charge = charge
        self.Zcharges = [8, 1, 1]

    def get_dipole_moment(self, dipole_matrix, density):
        return np.array([0.0, 0.0, 0.5])


class _FakeBasis:
    loaded = []

    @classmethod
    def load(cls, *, mol, basis_name):
        cls.loaded.append((mol, basis_name))
        return f"basis:{basis_name}"

    def __init__(self, mol, assignment):
        self.mol = mol
        self.assignment = assignment
        self.bfs_atoms = [0, 1, 2]


class _FakeDFT:
    last = None

    def __init__(self, mol, basis, auxbasis, **kwargs):
        type(self).last = self
        self.mol = mol
        self.basis = basis
        self.auxbasis = auxbasis
        self.kwargs = kwargs
        self.converged = True
        self.niter = 7
        self.mo_energies = [-0.8, -0.4, 0.1]
        self.mo_occupations = [2.0, 2.0, 0.0]
        self.mo_coefficients = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        self.max_itr = None
        self.sao = False

    def scf(self):
        print("fake PyFock SCF output")
        return -76.123456, np.diag([2.0, 1.0, 1.0])


class TestPyfockAdapter:
    def test_capabilities_include_phase2_analysis_and_geometry(self):
        caps = PyfockEngine().capabilities()
        assert caps.supported_calc_types == ("single_point", "geometry_opt")
        assert caps.supports_orbital_export is True

    def test_water_result_and_stdout_capture(self):
        stream = io.StringIO()
        request = _request(
            progress_stream=stream,
            options={"ncores": 4, "max_iterations": 30, "conv_crit": 1e-8},
        )
        fake_pyfock = ModuleType("pyfock")
        fake_pyfock.Integrals = type(
            "FakeIntegrals",
            (),
            {
                "overlap_mat_symm": staticmethod(lambda basis: np.eye(3)),
                "dipole_moment_mat_symm": staticmethod(
                    lambda basis: np.zeros((3, 3, 3))
                ),
            },
        )
        with (
            patch(
                "quantui.engines.pyfock_engine._load_pyfock_api",
                return_value=(_FakeMol, _FakeBasis, _FakeDFT),
            ),
            patch.dict(sys.modules, {"pyfock": fake_pyfock}),
        ):
            result = PyfockEngine().run(request)

        assert result.status == "success"
        assert result.engine_id == "pyfock"
        assert result.energy_hartree == pytest.approx(-76.123456)
        assert result.homo_lumo_gap_ev == pytest.approx(0.5 * 27.211386245988)
        assert result.n_iterations == 7
        assert result.native_result.engine_id == "pyfock"
        assert result.native_result.density_fit is True
        assert result.native_result.scf_variant == "RKS"
        assert result.native_result.mo_energy_hartree.tolist() == [-0.8, -0.4, 0.1]
        assert result.native_result.mo_coeff.shape == (3, 3)
        assert result.native_result.mulliken_charges == pytest.approx([6.0, 0.0, 0.0])
        assert result.native_result.dipole_moment_debye == pytest.approx(
            0.5 * 2.541746473
        )
        assert _FakeDFT.last.sao is True
        assert "fake PyFock SCF output" in stream.getvalue()
        assert "density fitting: on" in stream.getvalue()

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("calc_type", "frequency", "Single Point and Geometry Opt"),
            ("method", "B3LYP", "PBE only"),
            ("basis", "LANL2DZ", "def2-SVP"),
            ("charge", 1, "neutral molecules"),
            ("multiplicity", 2, "closed-shell singlets"),
            ("solvent", "Water", "not available"),
        ],
    )
    def test_unsupported_science_is_rejected_early(self, field, value, message):
        with pytest.raises(UnsupportedCapabilityError) as exc:
            PyfockEngine().run(_request(**{field: value}))
        assert message in exc.value.user_message

    def test_third_party_failure_becomes_error_result(self):
        class FailingDFT(_FakeDFT):
            def scf(self):
                raise RuntimeError("integral failure")

        with patch(
            "quantui.engines.pyfock_engine._load_pyfock_api",
            return_value=(_FakeMol, _FakeBasis, FailingDFT),
        ):
            result = PyfockEngine().run(_request())
        assert result.status == "error"
        assert result.error["code"] == "PYFOCK_CALCULATION_FAILED"

    def test_geometry_optimization_dispatches_through_shared_optimizer(self):
        native = type(
            "Native",
            (),
            {
                "converged": True,
                "energy_hartree": -1.1,
                "n_steps": 3,
                "method": "PBE",
                "basis": "def2-SVP",
                "formula": "H2O",
            },
        )()
        with patch("quantui.optimizer.optimize_geometry", return_value=native) as opt:
            result = PyfockEngine().run(_request(calc_type="geometry_opt"))

        assert result.native_result is native
        assert result.engine_id == "pyfock"
        assert result.n_iterations == 3
        assert opt.call_args.kwargs["engine_id"] == "pyfock"


@pytest.mark.pyfock
@pytest.mark.skipif(
    os.environ.get("QUANTUI_RUN_PYFOCK_INTEGRATION") != "1",
    reason="set QUANTUI_RUN_PYFOCK_INTEGRATION=1 for the real PyFock check",
)
def test_real_pyfock_water_pbe_def2_svp_matches_pyscf_reference():
    """Windows CI gate: real PyFock energy stays near a PySCF reference."""
    result = PyfockEngine().run(_request(options={"ncores": 2, "max_iterations": 50}))
    assert result.status == "success", result.error
    assert result.converged is True
    # Reference generated with PySCF/PBE/def2-SVP at this exact geometry.
    assert result.energy_hartree == pytest.approx(-76.272034156700, rel=1.0e-5)
