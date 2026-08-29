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


class TestSubmitCooldown:
    def test_default_cooldown(self, monkeypatch):
        monkeypatch.delenv("QUANTUI_SLURM_SUBMIT_COOLDOWN_S", raising=False)
        assert cfg.submit_cooldown_seconds() == cfg.SUBMIT_COOLDOWN_SECONDS

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("QUANTUI_SLURM_SUBMIT_COOLDOWN_S", "45")
        assert cfg.submit_cooldown_seconds() == 45

    def test_zero_disables_cooldown(self, monkeypatch):
        monkeypatch.setenv("QUANTUI_SLURM_SUBMIT_COOLDOWN_S", "0")
        assert cfg.submit_cooldown_seconds() == 0
