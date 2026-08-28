"""
Tests for cluster security validators (salvaged from legacy archive).
"""

import pytest

from quantui.backends.cluster_security import (
    check_concurrent_job_limit,
    safe_join,
    validate_email,
    validate_mail_events,
    validate_resources,
)
from quantui.security import SecurityError


class TestValidateResources:
    def test_valid_resources(self):
        out = validate_resources(4, 8, "04:00:00")
        assert out == {"cores": 4, "memory_gb": 8, "walltime": "04:00:00"}

    def test_rejects_excessive_cores(self):
        with pytest.raises(SecurityError, match="cores"):
            validate_resources(999, 8, "04:00:00")

    def test_rejects_bad_walltime(self):
        with pytest.raises(SecurityError, match="walltime"):
            validate_resources(4, 8, "bad")


class TestConcurrentJobLimit:
    def test_raises_at_limit(self):
        with pytest.raises(SecurityError, match="Concurrent job limit"):
            check_concurrent_job_limit(10, max_jobs=10)


class TestEmailValidation:
    def test_valid_email(self):
        assert validate_email("student@example.edu") == "student@example.edu"

    def test_invalid_email(self):
        with pytest.raises(SecurityError):
            validate_email("not-an-email")

    def test_mail_events_default(self):
        assert validate_mail_events(None) == ["END", "FAIL"]

    def test_unknown_mail_event(self):
        with pytest.raises(SecurityError):
            validate_mail_events(["FINISHED"])


class TestSafeJoin:
    def test_rejects_traversal(self, tmp_path):
        with pytest.raises(SecurityError):
            safe_join(tmp_path, "..", "etc", "passwd")

    def test_valid_join(self, tmp_path):
        path = safe_join(tmp_path, "job123.json")
        assert path.parent == tmp_path.resolve()
