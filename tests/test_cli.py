"""Tests for the ``quantui`` CLI (``quantui/cli.py``).

All tests are platform-independent. The CLI reads from
``~/.quantui/logs/event_log.jsonl`` by default, so each test overrides
``QUANTUI_LOG_DIR`` via ``monkeypatch`` to point at a ``tmp_path`` so we
never touch the real user log.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys

import pytest

from quantui import cli


@pytest.fixture
def isolated_log_dir(tmp_path, monkeypatch):
    """Point QuantUI's event log at a fresh tmp directory for one test."""
    monkeypatch.setenv("QUANTUI_LOG_DIR", str(tmp_path))
    return tmp_path


def _write_event_log(log_dir, events):
    path = log_dir / "event_log.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")
    return path


def _capture(argv):
    """Run cli.main with argv and return (exit_code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        rc = cli.main(argv)
    finally:
        sys.stdout, sys.stderr = real_out, real_err
    return rc, out.getvalue(), err.getvalue()


class TestLogTail:
    def test_missing_log_returns_zero_with_msg(self, isolated_log_dir):
        rc, out, err = _capture(["log", "tail"])
        assert rc == 0
        assert out == ""
        assert "no event log" in err

    def test_empty_log_returns_zero_with_msg(self, isolated_log_dir):
        _write_event_log(isolated_log_dir, [])
        rc, out, err = _capture(["log", "tail"])
        assert rc == 0
        assert out == ""
        assert "empty" in err

    def test_default_n_is_20(self, isolated_log_dir):
        events = [
            {
                "timestamp": f"2026-05-25T12:00:{i:02d}+00:00",
                "event": "tick",
                "message": f"msg-{i}",
            }
            for i in range(30)
        ]
        _write_event_log(isolated_log_dir, events)
        rc, out, _ = _capture(["log", "tail"])
        assert rc == 0
        # 20 lines printed; verify the LAST 20 are kept (msg-10..msg-29).
        lines = [ln for ln in out.splitlines() if ln.strip()]
        assert len(lines) == 20
        assert "msg-10" in lines[0]
        assert "msg-29" in lines[-1]

    def test_n_flag_overrides(self, isolated_log_dir):
        events = [
            {
                "timestamp": f"2026-05-25T12:00:{i:02d}+00:00",
                "event": "tick",
                "message": f"m{i}",
            }
            for i in range(10)
        ]
        _write_event_log(isolated_log_dir, events)
        rc, out, _ = _capture(["log", "tail", "-n", "3"])
        assert rc == 0
        lines = [ln for ln in out.splitlines() if ln.strip()]
        assert len(lines) == 3
        assert "m7" in lines[0]
        assert "m9" in lines[-1]

    def test_extras_appended_as_kv(self, isolated_log_dir):
        events = [
            {
                "timestamp": "2026-05-25T12:00:00+00:00",
                "event": "calc_done",
                "message": "B3LYP/STO-3G on H2O",
                "elapsed_ms": 4321,
                "gpu_used": True,
            },
        ]
        _write_event_log(isolated_log_dir, events)
        rc, out, _ = _capture(["log", "tail"])
        assert rc == 0
        # Both extras appear in k=v form.
        assert "elapsed_ms=4321" in out
        assert "gpu_used=True" in out
        # Core fields appear once.
        assert "calc_done" in out
        assert "B3LYP/STO-3G on H2O" in out


class TestCliParser:
    def test_no_args_exits_nonzero(self, isolated_log_dir):
        # argparse exits 2 when a required subparser is missing.
        with pytest.raises(SystemExit) as exc:
            _capture([])
        assert exc.value.code == 2

    def test_unknown_subcommand_exits_nonzero(self, isolated_log_dir):
        with pytest.raises(SystemExit) as exc:
            _capture(["bogus"])
        assert exc.value.code == 2

    def test_log_without_subcommand_exits_nonzero(self, isolated_log_dir):
        with pytest.raises(SystemExit) as exc:
            _capture(["log"])
        assert exc.value.code == 2


def test_fmt_event_renders_minimal_record():
    line = cli._fmt_event(
        {
            "timestamp": "2026-05-25T12:00:00+00:00",
            "event": "startup",
            "message": "QuantUI 0.2.0",
        }
    )
    assert "2026-05-25T12:00:00+00:00" in line
    assert "startup" in line
    assert "QuantUI 0.2.0" in line


def test_fmt_event_handles_missing_fields():
    # Should not raise even on a malformed record.
    line = cli._fmt_event({})
    assert "?" in line  # default event


class TestGpuCheck:
    """`quantui gpu check` — exit 0 when GPU available, 1 otherwise."""

    def test_disabled_via_env_var(self, monkeypatch, isolated_log_dir):
        monkeypatch.setenv("QUANTUI_DISABLE_GPU", "1")
        rc, out, err = _capture(["gpu", "check"])
        assert rc == 1
        assert "not available" in err
        assert "QUANTUI_DISABLE_GPU" in err

    def _patch_gpu4pyscf_import(self, monkeypatch, exc):
        """Make ``import gpu4pyscf`` raise *exc*, leaving other imports alone."""
        import builtins as _bi

        import quantui.gpu_offload as _gpuo

        _gpuo.is_gpu_available.cache_clear()
        _real_import = _bi.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "gpu4pyscf":
                raise exc
            return _real_import(name, *args, **kwargs)

        monkeypatch.setattr(_bi, "__import__", _fake_import)

    def test_reports_missing_gpu4pyscf(self, monkeypatch, isolated_log_dir):
        # Genuinely-absent package → ModuleNotFoundError.
        self._patch_gpu4pyscf_import(
            monkeypatch, ModuleNotFoundError("No module named 'gpu4pyscf'")
        )
        rc, out, err = _capture(["gpu", "check"])
        assert rc == 1
        assert "not installed" in err

    def test_broken_cuda_libs_not_reported_as_missing(
        self, monkeypatch, isolated_log_dir
    ):
        # GPU.7 regression: gpu4pyscf present but its CUDA libraries are not.
        # This is an ImportError, NOT a ModuleNotFoundError, and it used to be
        # reported as "gpu4pyscf not installed" — sending the user back to an
        # install step they had already done.
        self._patch_gpu4pyscf_import(
            monkeypatch, ImportError('Failure finding "libnvJitLink.so"')
        )
        rc, out, err = _capture(["gpu", "check"])
        assert rc == 1
        assert "not installed" not in err
        assert "failed to import" in err
        assert "libnvJitLink" in err

    def test_happy_path_when_gpu_detected(self, monkeypatch, isolated_log_dir):
        import quantui.gpu_offload as _gpuo

        # The CLI reads probe_gpu() for the (available, name, reason) triple.
        monkeypatch.setattr(_gpuo, "probe_gpu", lambda: (True, "NVIDIA H200", ""))
        rc, out, err = _capture(["gpu", "check"])
        assert rc == 0
        assert "GPU offload available" in out
        assert "NVIDIA H200" in out
        # Datacenter card — no FP64 advisory.
        assert "consumer" not in err

    def test_consumer_gpu_gets_fp64_advisory(self, monkeypatch, isolated_log_dir):
        # GPU.8: available is not the same as beneficial.
        import quantui.gpu_offload as _gpuo

        monkeypatch.setattr(
            _gpuo,
            "probe_gpu",
            lambda: (True, "NVIDIA GeForce RTX 5060 Ti", ""),
        )
        rc, out, err = _capture(["gpu", "check"])
        assert rc == 0
        assert "GPU offload available" in out
        assert "consumer" in err
        assert "SLOWER" in err


class TestAnalyticsBuild:
    """`quantui analytics build` — wraps analytics.build_dashboard."""

    def test_empty_perf_log_returns_zero_with_msg(self, isolated_log_dir):
        rc, out, err = _capture(["analytics", "build"])
        assert rc == 0
        assert "perf log is empty" in err

    def test_writes_file_at_explicit_path(self, isolated_log_dir, tmp_path):
        # Seed perf log so the dashboard has data.
        perf_path = isolated_log_dir / "perf_log.jsonl"
        perf_path.write_text(
            json.dumps(
                {
                    "timestamp": "2026-05-25T12:00:00+00:00",
                    "formula": "H2O",
                    "method": "B3LYP",
                    "basis": "STO-3G",
                    "elapsed_s": 1.0,
                    "converged": True,
                    "gpu_used": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        target = tmp_path / "report.html"
        rc, out, _ = _capture(["analytics", "build", "-o", str(target)])
        assert rc == 0
        assert target.exists()
        assert "Wrote" in out
        assert str(target) in out

    def _seed_perf_log(self, log_dir):
        """Helper: write one perf record so build_dashboard has data."""
        perf_path = log_dir / "perf_log.jsonl"
        perf_path.write_text(
            json.dumps(
                {
                    "timestamp": "2026-05-25T12:00:00+00:00",
                    "formula": "H2O",
                    "method": "B3LYP",
                    "basis": "STO-3G",
                    "elapsed_s": 1.0,
                    "converged": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def test_open_flag_calls_webbrowser_off_wsl(
        self, isolated_log_dir, tmp_path, monkeypatch
    ):
        # Force the non-WSL branch so the test runs the webbrowser path.
        monkeypatch.setattr(cli, "_is_wsl", lambda: False)
        self._seed_perf_log(isolated_log_dir)
        target = tmp_path / "report.html"

        opened_urls: list[str] = []
        import webbrowser as _wb

        def _fake_open(url, *_args, **_kwargs):
            opened_urls.append(url)
            return True

        monkeypatch.setattr(_wb, "open", _fake_open)

        rc, _, _ = _capture(["analytics", "build", "-o", str(target), "--open"])
        assert rc == 0
        assert target.exists()
        # The URL should be a file:// URI pointing at the written report.
        assert len(opened_urls) == 1
        assert opened_urls[0].startswith("file:")
        assert "report.html" in opened_urls[0]

    def test_open_flag_handles_browser_failure_gracefully(
        self, isolated_log_dir, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(cli, "_is_wsl", lambda: False)
        self._seed_perf_log(isolated_log_dir)
        target = tmp_path / "report.html"

        import webbrowser as _wb

        # Headless systems can return False from webbrowser.open.
        monkeypatch.setattr(_wb, "open", lambda *a, **k: False)

        rc, _, err = _capture(["analytics", "build", "-o", str(target), "--open"])
        # Exit code must remain 0 — the dashboard was written successfully.
        assert rc == 0
        assert "could not auto-open" in err


class TestWslAwareOpener:
    """`_open_in_browser` chooses wslview / explorer.exe on WSL."""

    def test_is_wsl_detects_env_var(self, monkeypatch):
        monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
        assert cli._is_wsl() is True

    def test_is_wsl_false_when_env_and_proc_missing(self, monkeypatch):
        # Both signals absent → must return False, not raise.
        monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
        import builtins

        original = builtins.open

        def _fail_open(*args, **kwargs):
            if args and args[0] == "/proc/version":
                raise OSError("simulated absence")
            return original(*args, **kwargs)

        monkeypatch.setattr(builtins, "open", _fail_open)
        assert cli._is_wsl() is False

    def test_wsl_prefers_wslview(self, monkeypatch, tmp_path):
        """On WSL, wslview is tried first and wins when it returns 0."""
        monkeypatch.setattr(cli, "_is_wsl", lambda: True)

        calls: list[list[str]] = []

        class _FakeRun:
            def __init__(self, returncode):
                self.returncode = returncode

        def _fake_subprocess_run(cmd, **_kwargs):
            calls.append(list(cmd))
            return _FakeRun(0)

        import subprocess

        monkeypatch.setattr(subprocess, "run", _fake_subprocess_run)
        target = tmp_path / "report.html"
        target.write_text("x", encoding="utf-8")

        ok, tool = cli._open_in_browser(target)
        assert ok is True
        assert tool == "wslview"
        assert len(calls) == 1
        assert calls[0][0] == "wslview"
        assert str(target) in calls[0]

    def test_wsl_falls_back_to_explorer_when_wslview_missing(
        self, monkeypatch, tmp_path
    ):
        """When wslview isn't installed (FileNotFoundError), explorer.exe runs."""
        monkeypatch.setattr(cli, "_is_wsl", lambda: True)

        calls: list[str] = []

        class _FakeRun:
            def __init__(self, returncode):
                self.returncode = returncode

        def _fake_subprocess_run(cmd, **_kwargs):
            tool = cmd[0]
            calls.append(tool)
            if tool == "wslview":
                raise FileNotFoundError("not installed")
            return _FakeRun(0)

        import subprocess

        monkeypatch.setattr(subprocess, "run", _fake_subprocess_run)
        target = tmp_path / "report.html"
        target.write_text("x", encoding="utf-8")

        ok, tool = cli._open_in_browser(target)
        assert ok is True
        assert tool == "explorer.exe"
        assert calls == ["wslview", "explorer.exe"]

    def test_wsl_returns_false_when_all_openers_fail(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli, "_is_wsl", lambda: True)

        import subprocess

        def _fake_run(cmd, **_kwargs):
            raise FileNotFoundError(f"{cmd[0]} not installed")

        monkeypatch.setattr(subprocess, "run", _fake_run)
        target = tmp_path / "report.html"
        target.write_text("x", encoding="utf-8")

        ok, tool = cli._open_in_browser(target)
        assert ok is False
        assert tool is None

    def test_non_wsl_uses_webbrowser(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli, "_is_wsl", lambda: False)

        opened: list[str] = []
        import webbrowser

        def _fake_open(url, *_args, **_kwargs):
            opened.append(url)
            return True

        monkeypatch.setattr(webbrowser, "open", _fake_open)
        target = tmp_path / "report.html"
        target.write_text("x", encoding="utf-8")

        ok, tool = cli._open_in_browser(target)
        assert ok is True
        assert tool == "webbrowser"
        assert opened[0].startswith("file:")


class TestCliAvoidsGuiStackImport:
    """M13 audit fix: ``import quantui.cli`` must not pull in ipywidgets.

    Importing any submodule of ``quantui`` always runs ``quantui/__init__.py``
    first (Python import semantics), and that module used to eagerly import
    ``QuantUIApp`` (and, transitively, ``ipywidgets``/``IPython`` widget
    machinery) even for callers that only want the pure-Python CLI helpers.
    This must run in a subprocess — checking ``sys.modules`` in-process is
    unreliable once other test modules in the same pytest session have
    already imported ipywidgets/quantui.app themselves.
    """

    def test_import_cli_does_not_load_ipywidgets_or_app(self):
        script = (
            "import sys\n"
            "import quantui.cli\n"
            "assert 'ipywidgets' not in sys.modules, 'ipywidgets was imported'\n"
            "assert 'quantui.app' not in sys.modules, 'quantui.app was imported'\n"
            "assert 'quantui.progress' not in sys.modules, "
            "'quantui.progress was imported'\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "OK" in result.stdout

    def test_lazy_attrs_still_resolve_on_demand(self):
        script = (
            "import quantui\n"
            "assert quantui.StepProgress(['a']).widget is not None\n"
            "assert len(quantui.HELP_TOPICS) > 0\n"
            "assert 'charge' in quantui.VALID_TOPICS\n"
            "quantui.help_panel('charge')\n"
            "assert quantui.QuantUIApp.__name__ == 'QuantUIApp'\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "OK" in result.stdout

    def test_unknown_attribute_still_raises(self):
        import quantui

        assert hasattr(quantui, "THIS_ATTR_DOES_NOT_EXIST") is False
