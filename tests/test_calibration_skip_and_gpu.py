"""Tests for the session-55 calibration UX fixes:

1. **Skip button**: replaces the per-step timeout. The user can abandon
   ONE step without losing the whole calibration (the old hard 1800 s
   tier-4 cap cut off a near-finishing benzene B3LYP/6-31G* freq).
2. **MP2 + CCSD blocked on GPU**: gpu4pyscf's post-HF support is
   experimental and was crashing immediately after the RHF reference.
   Both methods now stay CPU-side via ``_GPU_UNSUPPORTED_METHODS``.
3. **error_msg visible in calibration table**: failed steps now show
   the captured error message inline (truncated) so the user knows
   WHY a step failed.

All tests platform-independent. No PySCF required.
"""

from __future__ import annotations

import inspect

# =====================================================================
# Fix 2 — MP2 + CCSD on the GPU skip list
# =====================================================================


class TestGpuUnsupportedMethods:
    def test_mp2_blocked_on_gpu(self):
        from quantui.gpu_offload import _GPU_UNSUPPORTED_METHODS

        assert "MP2" in _GPU_UNSUPPORTED_METHODS

    def test_ccsd_blocked_on_gpu(self):
        from quantui.gpu_offload import _GPU_UNSUPPORTED_METHODS

        assert "CCSD" in _GPU_UNSUPPORTED_METHODS

    def test_ccsd_t_still_blocked(self):
        # Don't accidentally remove the original entry while adding new ones.
        from quantui.gpu_offload import _GPU_UNSUPPORTED_METHODS

        assert "CCSD(T)" in _GPU_UNSUPPORTED_METHODS

    def test_try_to_gpu_returns_cpu_path_for_mp2(self):
        # Direct functional check: try_to_gpu should short-circuit before
        # calling .to_gpu() when the method is blocked. The "mf" we pass
        # doesn't need to be real — try_to_gpu returns it unchanged.
        from quantui.gpu_offload import try_to_gpu

        sentinel = object()
        mf, used_gpu, name = try_to_gpu(sentinel, "MP2")
        assert mf is sentinel
        assert used_gpu is False
        assert name is None


# =====================================================================
# Fix 1 — Skip event + no-timeout default
# =====================================================================


class TestRunCalibrationSignature:
    def test_run_calibration_accepts_skip_event(self):
        from quantui.benchmarks import run_calibration

        sig = inspect.signature(run_calibration)
        assert "skip_event" in sig.parameters

    def test_timeout_per_step_default_is_none(self):
        # session 55 user request: no automatic timeout — Skip button
        # is the user-facing control.
        from quantui.benchmarks import run_calibration

        sig = inspect.signature(run_calibration)
        timeout_param = sig.parameters["timeout_per_step"]
        assert timeout_param.default is None

    def test_loop_handles_none_timeout_without_crashing(self):
        # Most direct path: run_calibration with PySCF unavailable just
        # iterates through the suite emitting PySCF-not-available errors.
        # With timeout_per_step=None we must NOT hit the
        # ``elapsed > timeout_per_step`` comparison (which would
        # TypeError on None).
        from quantui.benchmarks import run_calibration

        # Smaller suite so the test stays fast.
        result = run_calibration(mode="tier1", timeout_per_step=None)
        # On Windows (no PySCF) every step is marked error.
        # Function returns cleanly without exceptions.
        assert result.mode == "tier1"

    def test_skipped_status_constant_exists(self):
        from quantui import benchmarks

        assert hasattr(benchmarks, "_STATUS_SKIPPED")
        assert benchmarks._STATUS_SKIPPED == "skipped"


class TestSkipEventInPollLoop:
    """Structural / source check: the poll loop now honours skip_event.

    A full end-to-end skip test would require PySCF + spawning a real
    worker; the source-grep test is the cheap regression guard.
    """

    def test_poll_loop_checks_skip_event(self):
        from quantui import benchmarks

        src = inspect.getsource(benchmarks.run_calibration)
        # The new branch checks skip_event.is_set() and calls
        # skip_event.clear() so the next step starts fresh.
        assert "skip_event" in src
        assert "skip_event.is_set()" in src
        assert "skip_event.clear()" in src
        assert "_STATUS_SKIPPED" in src

    def test_no_unconditional_timeout_comparison(self):
        # If someone reintroduces ``elapsed > timeout_per_step`` without
        # a None guard, this test catches it.
        from quantui import benchmarks

        src = inspect.getsource(benchmarks.run_calibration)
        # Either the comparison is guarded by a None check OR it's gone.
        # Match the guard pattern explicitly.
        assert "timeout_per_step is not None" in src


# =====================================================================
# Fix 3 — error_msg surfaced in the table
# =====================================================================


class TestCalTableShowsErrorMsg:
    def test_error_row_includes_error_msg_text(self):
        # Direct render-helper test: an error step should include the
        # error_msg in the rendered HTML so users see WHY the step failed.
        from types import SimpleNamespace

        from quantui.app_runflow import _cal_table_html

        bad_step = SimpleNamespace(
            label="H₂O MP2/cc-pVDZ",
            method="MP2",
            basis="cc-pVDZ",
            n_atoms=3,
            n_electrons=10,
            n_basis=24,
            status="error",
            elapsed_s=5.54,
            error_msg="MP2 correction failed for H2O: foo bar baz",
            calc_type="single_point",
            result_dir=None,
        )
        html = _cal_table_html([bad_step], total=1)
        assert "✗ error" in html
        # The error message text appears in the rendered HTML.
        assert "MP2 correction failed" in html

    def test_ok_row_does_not_show_inline_detail(self):
        from types import SimpleNamespace

        from quantui.app_runflow import _cal_table_html

        good_step = SimpleNamespace(
            label="H₂ RHF/STO-3G",
            method="RHF",
            basis="STO-3G",
            n_atoms=2,
            n_electrons=2,
            n_basis=2,
            status="ok",
            elapsed_s=0.5,
            error_msg="",
            calc_type="single_point",
            result_dir=None,
        )
        html = _cal_table_html([good_step], total=1)
        # No italic detail line for successful steps.
        assert "font-style:italic" not in html or "color:#94a3b8" not in html

    def test_long_error_msg_truncated(self):
        from types import SimpleNamespace

        from quantui.app_runflow import _cal_table_html

        long_msg = "x" * 500
        bad_step = SimpleNamespace(
            label="bad",
            method="MP2",
            basis="cc-pVDZ",
            n_atoms=3,
            n_electrons=10,
            n_basis=24,
            status="error",
            elapsed_s=1.0,
            error_msg=long_msg,
            calc_type="single_point",
            result_dir=None,
        )
        html = _cal_table_html([bad_step], total=1)
        # The 500-char message gets truncated with "…".
        assert "…" in html
        # And isn't dumped wholesale (would be > 200 chars of x's).
        assert "x" * 200 not in html

    def test_skipped_row_uses_skipped_label(self):
        from types import SimpleNamespace

        from quantui.app_runflow import _cal_status_text, _cal_table_html

        # Direct check of the status renderer.
        assert "skipped" in _cal_status_text("skipped").lower()

        skipped_step = SimpleNamespace(
            label="C₆H₆ B3LYP [Freq]",
            method="B3LYP",
            basis="6-31G*",
            n_atoms=12,
            n_electrons=42,
            n_basis=96,
            status="skipped",
            elapsed_s=1500.0,
            error_msg="skipped by user at 1500s",
            calc_type="frequency",
            result_dir=None,
        )
        html = _cal_table_html([skipped_step], total=1)
        assert "⏭" in html or "skipped" in html


# =====================================================================
# UI wiring — Skip button + handler exist
# =====================================================================


class TestSkipButtonWiring:
    def test_app_has_cal_skip_btn(self):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        assert hasattr(app, "_cal_skip_btn")

    def test_app_has_on_cal_skip_method(self):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        assert callable(getattr(app, "_on_cal_skip", None))

    def test_on_cal_skip_handler_in_app_runflow(self):
        from quantui import app_runflow

        assert callable(getattr(app_runflow, "on_cal_skip", None))
