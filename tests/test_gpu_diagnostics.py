"""Tests for GPU probe diagnostics (GPU.7) and the FP64 advisory (GPU.8).

Background — GPU.7 (found during the 2026-07-29 local WSL validation): a
gpu4pyscf install whose CUDA math libraries were missing failed with
``ImportError: Failure finding "libnvJitLink.so"``, and QuantUI reported it as
"gpu4pyscf not installed", pointing the user back at an install step they had
already completed. ``ModuleNotFoundError`` is a *subclass* of ``ImportError``,
so the two cases have to be caught separately to be told apart.

Background — GPU.8: consumer GPUs gate FP64 to ~1/32-1/64 of FP32 while PySCF
SCF is FP64 throughout, so offload on such a card can be slower than the CPU
(measured: RTX 5060 Ti at 0.44x a 20-core CPU). Detection must therefore be able
to say "available, but probably not worth it".

Platform-independent: no GPU, no CUDA, no PySCF.
"""

from __future__ import annotations

import builtins

import pytest

from quantui import gpu_offload
from quantui.gpu_offload import (
    is_gpu_available,
    is_low_fp64_device,
    probe_gpu,
)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Fresh probe + isolated settings for every test."""
    monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.delenv("QUANTUI_DISABLE_GPU", raising=False)
    is_gpu_available.cache_clear()
    yield
    is_gpu_available.cache_clear()


def _fail_gpu4pyscf_import(monkeypatch, exc: BaseException):
    """Make ``import gpu4pyscf`` raise *exc*, leaving other imports alone."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "gpu4pyscf":
            raise exc
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


class TestProbeDistinguishesImportFailures:
    """GPU.7 — "absent" and "installed but broken" must not share a message."""

    def test_module_not_found_reports_not_installed(self, monkeypatch):
        _fail_gpu4pyscf_import(
            monkeypatch, ModuleNotFoundError("No module named 'gpu4pyscf'")
        )
        available, name, reason = probe_gpu()

        assert available is False
        assert name is None
        assert "not installed" in reason

    def test_broken_import_chain_is_not_reported_as_not_installed(self, monkeypatch):
        # The exact failure seen on the WSL desktop: package present, CUDA
        # libraries missing. This is an ImportError but NOT a
        # ModuleNotFoundError.
        _fail_gpu4pyscf_import(
            monkeypatch, ImportError('Failure finding "libnvJitLink.so"')
        )
        available, name, reason = probe_gpu()

        assert available is False
        # The regression: this used to say "not installed".
        assert "not installed" not in reason
        assert "failed to import" in reason
        # The real cause must reach the user, not be swallowed.
        assert "libnvJitLink" in reason

    def test_broken_import_chain_is_logged(self, monkeypatch, caplog):
        # The old code's ImportError branch logged nothing at all, so
        # `quantui log tail` could not reveal the cause either.
        _fail_gpu4pyscf_import(
            monkeypatch, ImportError('Failure finding "libnvJitLink.so"')
        )
        with caplog.at_level("WARNING", logger=gpu_offload.__name__):
            probe_gpu()

        assert any("failed to import" in r.message for r in caplog.records)

    def test_unexpected_exception_type_is_surfaced(self, monkeypatch):
        _fail_gpu4pyscf_import(monkeypatch, RuntimeError("driver exploded"))
        available, _name, reason = probe_gpu()

        assert available is False
        assert "RuntimeError" in reason
        assert "driver exploded" in reason

    def test_env_disabled_reports_env_reason(self, monkeypatch):
        monkeypatch.setenv("QUANTUI_DISABLE_GPU", "1")
        is_gpu_available.cache_clear()
        available, _name, reason = probe_gpu()

        assert available is False
        assert "QUANTUI_DISABLE_GPU" in reason


class TestIsGpuAvailableStillWorks:
    """The 2-tuple API and its cache_clear must survive the refactor."""

    def test_returns_two_tuple(self):
        result = is_gpu_available()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_cache_clear_actually_clears(self, monkeypatch):
        # The cache moved onto the private probe; if cache_clear were not
        # forwarded, this env change would be invisible and the assert below
        # would see a stale result.
        first = is_gpu_available()
        monkeypatch.setenv("QUANTUI_DISABLE_GPU", "1")
        is_gpu_available.cache_clear()
        assert is_gpu_available() == (False, None)
        # sanity: the call above genuinely re-probed rather than replaying
        assert first is not None

    def test_agrees_with_probe_gpu(self):
        available, name, _reason = probe_gpu()
        assert is_gpu_available() == (available, name)


class TestSettingsToggle:
    """GPU.8 — the persistent preference gates offload."""

    def test_disabled_in_settings_blocks_gpu(self):
        from quantui.user_settings import UserSettings

        s = UserSettings.load()
        s.compute.gpu_enabled = False
        s.save()
        is_gpu_available.cache_clear()

        available, _name, reason = probe_gpu()
        assert available is False
        assert "settings" in reason.lower()

    def test_enabled_by_default(self):
        from quantui.user_settings import UserSettings

        assert UserSettings.load().compute.gpu_enabled is True

    def test_setting_round_trips(self):
        from quantui.user_settings import UserSettings

        s = UserSettings.load()
        s.compute.gpu_enabled = False
        s.save()
        assert UserSettings.load().compute.gpu_enabled is False

    def test_unreadable_settings_do_not_disable_gpu(self, monkeypatch):
        # A broken settings file must never be the thing that turns off the GPU.
        def boom(*_a, **_k):
            raise OSError("settings unreadable")

        monkeypatch.setattr(
            "quantui.user_settings.UserSettings.load", staticmethod(boom)
        )
        assert gpu_offload._gpu_enabled_in_settings() is True


class TestLowFp64Detection:
    """GPU.8 — advisory classification by device name."""

    @pytest.mark.parametrize(
        "name",
        [
            "NVIDIA GeForce RTX 5060 Ti",
            "NVIDIA GeForce GTX 1080",
            "NVIDIA RTX A6000",
            "Quadro P2000",
            "NVIDIA L40S",
            "Tesla T4",
        ],
    )
    def test_consumer_and_inference_cards_are_low_fp64(self, name):
        assert is_low_fp64_device(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "NVIDIA H200",
            "NVIDIA H100 80GB HBM3",
            "NVIDIA A100-SXM4-40GB",
            "Tesla V100-SXM2-16GB",
            "NVIDIA GH200 480GB",
            "Tesla P100-PCIE-16GB",
            "NVIDIA A30",
        ],
    )
    def test_datacenter_cards_are_not_low_fp64(self, name):
        assert is_low_fp64_device(name) is False

    def test_unknown_device_defaults_to_low_fp64(self):
        # Deliberate: consumer hardware is the common student case, so an
        # unrecognised name warns rather than silently promising speed.
        assert is_low_fp64_device("Some Future GPU 9000") is True

    def test_no_name_is_not_flagged(self):
        assert is_low_fp64_device(None) is False
        assert is_low_fp64_device("") is False

    def test_case_insensitive(self):
        assert is_low_fp64_device("nvidia h200") is False
        assert is_low_fp64_device("nvidia geforce rtx 4090") is True
