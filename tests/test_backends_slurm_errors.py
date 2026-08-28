"""
Tests for quantui.backends.slurm_errors (salvaged legacy module).
"""

import pytest

from quantui.backends.slurm_errors import (
    ErrorTranslation,
    format_error_for_student,
    format_error_html,
    translate_slurm_error,
)


class TestTranslateSlurmError:
    def test_none_on_empty_input(self):
        assert translate_slurm_error("") is None
        assert translate_slurm_error("   ") is None

    def test_qos_max_jobs(self):
        stderr = "sbatch: error: QOSMaxSubmitJobPerUserLimit reached"
        t = translate_slurm_error(stderr)
        assert t is not None
        assert t.category == "Job Limit Reached"
        assert isinstance(t, ErrorTranslation)

    def test_scf_not_converged(self):
        t = translate_slurm_error("RuntimeWarning: SCF not converged")
        assert t is not None
        assert "converge" in t.category.lower()

    def test_case_insensitive_matching(self):
        t = translate_slurm_error("scf not converged")
        assert t is not None


class TestFormatErrorForStudent:
    def test_known_error_includes_advice(self):
        msg = format_error_for_student("SCF not converged")
        assert "What to do" in msg

    def test_unknown_error_shows_raw_text(self):
        msg = format_error_for_student("weird error nobody expected")
        assert "weird error nobody expected" in msg


class TestFormatErrorHtml:
    def test_known_error_returns_html(self):
        html = format_error_html("QOSMaxSubmitJobPerUserLimit")
        assert "<div" in html
        assert "Job Limit Reached" in html
