"""Tests for cluster configuration helpers."""


from quantui.backends import cluster_config as cfg


class TestMaxConcurrentJobs:
    def test_default_limit(self, monkeypatch):
        monkeypatch.delenv("QUANTUI_MAX_CONCURRENT_JOBS", raising=False)
        assert cfg.max_concurrent_jobs() == cfg.MAX_CONCURRENT_JOBS

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("QUANTUI_MAX_CONCURRENT_JOBS", "5")
        assert cfg.max_concurrent_jobs() == 5

    def test_env_override_minimum_one(self, monkeypatch):
        monkeypatch.setenv("QUANTUI_MAX_CONCURRENT_JOBS", "0")
        assert cfg.max_concurrent_jobs() == 1

    def test_invalid_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("QUANTUI_MAX_CONCURRENT_JOBS", "not-a-number")
        assert cfg.max_concurrent_jobs() == cfg.MAX_CONCURRENT_JOBS
