"""
Tests for SLURM dispatch helpers and app integration utilities.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from quantui.backends.base import CalculationRequest
from quantui.backends.dispatch import (
    build_calculation_request,
    calc_type_key_from_app,
    is_slurm_available,
)
from quantui.app_slurm import use_slurm_execution


def _fake_app(**overrides):
    mol = SimpleNamespace(
        atoms=["H", "H"],
        coordinates=[[0, 0, 0], [0, 0, 0.74]],
        charge=0,
        get_formula=lambda: "H2",
    )
    defaults = dict(
        _molecule=mol,
        method_dd=SimpleNamespace(value="RHF"),
        basis_dd=SimpleNamespace(value="STO-3G"),
        mult_si=SimpleNamespace(value=1),
        calc_type_dd=SimpleNamespace(value="Single Point"),
        solvent_dd=SimpleNamespace(value=""),
        solvent_cb=SimpleNamespace(value=False),
        _user_settings=SimpleNamespace(
            compute=SimpleNamespace(execution_backend="local")
        ),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestDispatch:
    def test_build_calculation_request(self):
        req = build_calculation_request(_fake_app(), request_id="abc")
        assert req.request_id == "abc"
        assert req.calc_type == "single_point"
        assert req.molecule["atoms"] == ["H", "H"]

    def test_calc_type_key_from_app(self):
        app = _fake_app(calc_type_dd=SimpleNamespace(value="Frequency"))
        assert calc_type_key_from_app(app) == "frequency"

    @patch("quantui.backends.dispatch.shutil.which")
    def test_is_slurm_available(self, mock_which):
        mock_which.return_value = "/usr/bin/sbatch"
        assert is_slurm_available() is True
        mock_which.return_value = None
        assert is_slurm_available() is False


class TestUseSlurmExecution:
    @patch("quantui.app_slurm.is_slurm_available", return_value=True)
    def test_true_when_pref_slurm(self, _mock):
        app = _fake_app(
            _user_settings=SimpleNamespace(
                compute=SimpleNamespace(execution_backend="slurm")
            )
        )
        assert use_slurm_execution(app) is True

    @patch("quantui.app_slurm.is_slurm_available", return_value=False)
    def test_false_when_no_sbatch(self, _mock):
        app = _fake_app(
            _user_settings=SimpleNamespace(
                compute=SimpleNamespace(execution_backend="slurm")
            )
        )
        assert use_slurm_execution(app) is False

    @patch("quantui.app_slurm.is_slurm_available", return_value=True)
    def test_false_when_pref_local(self, _mock):
        assert use_slurm_execution(_fake_app()) is False
