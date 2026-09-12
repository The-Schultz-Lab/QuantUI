"""Quantum engine registry (M-PYFOCK PYF.1).

Tests lazy engine probes, capability handshakes, and resolve policy without
requiring PyFock to be installed. PySCF is present in CI/Linux cloud sessions.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from quantui.engines import (
    DEFAULT_ENGINE_PREFERENCE,
    VALID_ENGINE_PREFERENCES,
    EngineUnavailableError,
    PyfockEngine,
    PyscfEngine,
    build_pyscf_engine,
    is_pyscf_available,
    list_engines,
    resolve_engine,
    run_calc,
)
from quantui.engines.base import EngineCapabilities, EngineRequest, EngineResult
from quantui.user_settings import UserSettings


class TestLazyImport:
    def test_import_engines_without_pyfock_installed(self):
        import quantui.engines as engines_mod

        assert hasattr(engines_mod, "resolve_engine")

    def test_is_pyscf_available_matches_build(self):
        assert is_pyscf_available() == (build_pyscf_engine() is not None)


class TestPyscfCapabilities:
    @pytest.mark.skipif(not is_pyscf_available(), reason="PySCF not installed")
    def test_handshake_includes_full_calc_menu(self):
        caps = PyscfEngine().capabilities()
        assert caps.engine_id == "pyscf"
        assert "geometry_opt" in caps.supported_calc_types
        assert "B3LYP" in caps.supported_methods
        assert caps.supports_post_hf is True
        assert caps.supports_orbital_export is True


class TestPyfockCapabilities:
    def test_handshake_is_phase1_subset(self):
        caps = PyfockEngine().capabilities()
        assert caps.engine_id == "pyfock"
        assert caps.supported_calc_types == ("single_point",)
        assert caps.supported_methods == ("PBE",)
        assert caps.supported_basis_sets == ("def2-SVP", "def2-TZVP")
        assert caps.recommended_auxbasis == "def2-universal-jfit"
        assert caps.supports_post_hf is False


class TestRegistryProbes:
    def test_list_engines_mocked_both(self):
        with (
            patch("quantui.engines.is_pyscf_available", return_value=True),
            patch("quantui.engines.is_pyfock_available", return_value=True),
            patch("quantui.engines.build_pyscf_engine", return_value=PyscfEngine()),
            patch("quantui.engines.build_pyfock_engine", return_value=PyfockEngine()),
        ):
            caps = list_engines()
        assert {c.engine_id for c in caps} == {"pyscf", "pyfock"}

    def test_list_engines_mocked_pyfock_only(self):
        with (
            patch("quantui.engines.build_pyscf_engine", return_value=None),
            patch("quantui.engines.build_pyfock_engine", return_value=PyfockEngine()),
        ):
            caps = list_engines()
        assert [c.engine_id for c in caps] == ["pyfock"]

    def test_resolve_auto_prefers_pyscf_when_both_installed(self):
        with (
            patch("quantui.engines.build_pyscf_engine", return_value=PyscfEngine()),
            patch("quantui.engines.build_pyfock_engine", return_value=PyfockEngine()),
        ):
            engine = resolve_engine("auto")
        assert engine.engine_id == "pyscf"

    def test_resolve_auto_pyfock_only(self):
        with (
            patch("quantui.engines.build_pyscf_engine", return_value=None),
            patch("quantui.engines.build_pyfock_engine", return_value=PyfockEngine()),
        ):
            engine = resolve_engine("auto")
        assert engine.engine_id == "pyfock"

    def test_resolve_auto_windows_pyfock_only(self):
        with (
            patch.object(sys, "platform", "win32"),
            patch("quantui.engines.build_pyscf_engine", return_value=None),
            patch("quantui.engines.build_pyfock_engine", return_value=PyfockEngine()),
        ):
            engine = resolve_engine("auto")
        assert engine.engine_id == "pyfock"

    def test_resolve_explicit_pyfock(self):
        with (
            patch("quantui.engines.build_pyscf_engine", return_value=PyscfEngine()),
            patch("quantui.engines.build_pyfock_engine", return_value=PyfockEngine()),
        ):
            engine = resolve_engine("pyfock")
        assert engine.engine_id == "pyfock"

    def test_resolve_missing_engine_raises(self):
        with (
            patch("quantui.engines.build_pyscf_engine", return_value=None),
            patch("quantui.engines.build_pyfock_engine", return_value=None),
        ):
            with pytest.raises(EngineUnavailableError) as exc:
                resolve_engine("auto")
        assert exc.value.code == "ENGINE_UNAVAILABLE"

    def test_resolve_unavailable_preference_raises(self):
        with (
            patch("quantui.engines.build_pyscf_engine", return_value=PyscfEngine()),
            patch("quantui.engines.build_pyfock_engine", return_value=None),
        ):
            with pytest.raises(EngineUnavailableError):
                resolve_engine("pyfock")


class TestSettingsIntegration:
    def test_default_quantum_engine_is_auto(self):
        assert UserSettings().compute.quantum_engine == DEFAULT_ENGINE_PREFERENCE

    def test_valid_preferences_constant(self):
        assert VALID_ENGINE_PREFERENCES == ("auto", "pyscf", "pyfock")

    @pytest.mark.parametrize("engine", ["auto", "pyscf", "pyfock"])
    def test_quantum_engine_round_trips(self, tmp_path, engine):
        path = tmp_path / "settings.json"
        settings = UserSettings()
        settings.compute.quantum_engine = engine
        settings.save(path)
        loaded = UserSettings.load(path)
        assert loaded.compute.quantum_engine == engine

    def test_invalid_quantum_engine_falls_back(self, tmp_path):
        import json

        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps(
                {
                    "_schema_version": 1,
                    "compute": {"quantum_engine": "psi4"},
                }
            )
        )
        loaded = UserSettings.load(path)
        assert loaded.compute.quantum_engine == "auto"


class TestEngineCapabilitiesFrozen:
    def test_capabilities_is_immutable(self):
        caps = EngineCapabilities(
            engine_id="demo",
            display_name="Demo",
            supported_calc_types=("single_point",),
            supported_methods=("PBE",),
            supported_basis_sets=None,
            supports_solvent=False,
            supports_checkpoint_warm_start=False,
            supports_gpu=False,
            supports_post_hf=False,
            supports_orbital_export=False,
        )
        with pytest.raises(AttributeError):
            caps.engine_id = "other"  # type: ignore[misc]


class TestDispatch:
    def test_run_calc_calls_resolved_engine(self):
        request = EngineRequest(
            request_id="req-1",
            calc_type="single_point",
            method="PBE",
            basis="def2-SVP",
            charge=0,
            multiplicity=1,
            molecule={"atoms": ["He"], "coordinates": [[0.0, 0.0, 0.0]]},
        )
        expected = EngineResult(
            request_id="req-1",
            engine_id="pyfock",
            status="success",
            converged=True,
            energy_hartree=-2.0,
            method="PBE",
            basis="def2-SVP",
            formula="He",
        )
        engine = PyfockEngine()
        with (
            patch("quantui.engines.resolve_engine", return_value=engine),
            patch.object(engine, "run", return_value=expected) as mocked_run,
        ):
            actual = run_calc(request, "pyfock")
        assert actual is expected
        mocked_run.assert_called_once_with(request)

    def test_request_portable_dict_excludes_runtime_objects(self):
        stream = object()
        checkpoint = object()
        request = EngineRequest(
            request_id="req-1",
            calc_type="single_point",
            method="PBE",
            basis="def2-SVP",
            charge=0,
            multiplicity=1,
            molecule={"atoms": ["He"], "coordinates": [[0.0, 0.0, 0.0]]},
            progress_stream=stream,  # type: ignore[arg-type]
            checkpoint=checkpoint,
        )
        payload = request.to_dict()
        assert "progress_stream" not in payload
        assert "checkpoint" not in payload
        assert EngineRequest.from_dict(payload).warm_start is True
