"""Viewer mode: History + Analysis only, pointed at a results folder."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from quantui import app_launcher, cli
from quantui.app import QuantUIApp
from quantui.app_viewer import RESULTS_DIR_ENV, resolve_results_folder
from quantui.results_storage import save_result


def _save(results_dir, formula="H2O"):
    res = SimpleNamespace(
        formula=formula,
        method="RHF",
        basis="STO-3G",
        energy_hartree=-74.96,
        converged=True,
        n_iterations=7,
        homo_lumo_gap_ev=None,
    )
    return save_result(res, results_dir=results_dir)


@pytest.fixture
def results_env(tmp_path, monkeypatch):
    # setenv (not delenv) so monkeypatch restores the variable even after
    # viewer code rewrites os.environ directly.
    monkeypatch.setenv(RESULTS_DIR_ENV, str(tmp_path))
    return tmp_path


class TestResolveResultsFolder:
    def test_empty(self):
        assert resolve_results_folder("  ") == (None, "Enter a folder path.")

    def test_missing(self, tmp_path):
        path, reason = resolve_results_folder(str(tmp_path / "nope"))
        assert path is None and "does not exist" in reason

    def test_strips_quotes(self, tmp_path):
        path, _ = resolve_results_folder(f'"{tmp_path}"')
        assert path == tmp_path.resolve()

    def test_single_result_dir_maps_to_parent(self, tmp_path):
        rd = _save(tmp_path)
        path, _ = resolve_results_folder(str(rd))
        assert path == tmp_path.resolve()


class TestViewerApp:
    def test_tabs_are_history_and_analysis(self, results_env, monkeypatch):
        monkeypatch.setenv(RESULTS_DIR_ENV, str(results_env))
        app = QuantUIApp(viewer=True)
        titles = [app.root_tab.get_title(i) for i in range(len(app.root_tab.children))]
        assert titles == ["History", "Analysis"]
        assert app._tab_index("analysis") == 1

    def test_full_app_unchanged(self, results_env, monkeypatch):
        monkeypatch.setenv(RESULTS_DIR_ENV, str(results_env))
        app = QuantUIApp()
        assert app._root_tab_order()[0] == "calculate"
        assert not hasattr(app, "_viewer_folder_bar")

    def test_results_dir_argument_lists_results(self, results_env, monkeypatch):
        _save(results_env)
        app = QuantUIApp(viewer=True, results_dir=results_env)
        paths = [v for _, v in app.past_dd.options if v]
        assert len(paths) == 1

    def test_open_folder_switches_history(self, results_env, monkeypatch, tmp_path):
        monkeypatch.setenv(RESULTS_DIR_ENV, str(results_env))
        app = QuantUIApp(viewer=True)
        assert not [v for _, v in app.past_dd.options if v]
        other = tmp_path / "other"
        other.mkdir()
        _save(other)
        _save(other, formula="NH3")
        app._viewer_folder_txt.value = str(other)
        app._viewer_open_btn.click()
        assert len([v for _, v in app.past_dd.options if v]) == 2
        assert "2 calculations found" in app._viewer_status_html.value

    def test_open_folder_bad_path_reports_error(self, results_env, monkeypatch):
        monkeypatch.setenv(RESULTS_DIR_ENV, str(results_env))
        app = QuantUIApp(viewer=True)
        app._viewer_folder_txt.value = str(results_env / "missing")
        app._viewer_open_btn.click()
        assert "does not exist" in app._viewer_status_html.value


class TestViewerLauncher:
    def test_viewer_notebook_uses_viewer_mode(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUANTUI_HOME", str(tmp_path))
        nb = json.loads(app_launcher.ensure_viewer_notebook().read_text())
        src = "".join(nb["cells"][1]["source"])
        assert "QuantUIApp(viewer=True).display()" in src
        # The student app notebook is untouched.
        assert "viewer" not in "".join(app_launcher._APP_NOTEBOOK["cells"][1]["source"])

    def test_find_free_port(self):
        assert 0 < app_launcher.find_free_port() < 65536

    def test_missing_folder_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app_launcher, "voila_executable", lambda: "voila")
        assert app_launcher.run_viewer_app(tmp_path / "missing") == 1

    def test_cli_view_parses(self, monkeypatch, tmp_path):
        seen = {}

        def fake_run(folder, *, port, open_browser):
            seen.update(folder=folder, port=port, open_browser=open_browser)
            return 0

        monkeypatch.setattr(app_launcher, "run_viewer_app", fake_run)
        assert cli.main(["view", str(tmp_path), "--no-browser", "--port", "9001"]) == 0
        assert seen == {"folder": tmp_path, "port": 9001, "open_browser": False}


class TestExtractResultsZip:
    @staticmethod
    def _zip(files):
        import io
        import zipfile

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for name, data in files.items():
                zf.writestr(name, data)
        return buf.getvalue()

    def test_wrapper_folder_is_unwrapped(self, tmp_path):
        from quantui.app_viewer import extract_results_zip

        data = self._zip({"results/run1/result.json": "{}"})
        folder = extract_results_zip(data, "results.zip", root=tmp_path)
        assert folder.name == "results"
        assert (folder / "run1" / "result.json").is_file()

    def test_flat_archive_returns_destination(self, tmp_path):
        from quantui.app_viewer import extract_results_zip

        data = self._zip({"run1/result.json": "{}", "run2/result.json": "{}"})
        folder = extract_results_zip(data, "mine.zip", root=tmp_path)
        assert folder == (tmp_path / "mine").resolve()

    def test_path_traversal_rejected(self, tmp_path):
        from quantui.app_viewer import extract_results_zip

        data = self._zip({"../evil.txt": "x"})
        with pytest.raises(ValueError, match="Unsafe path"):
            extract_results_zip(data, "bad.zip", root=tmp_path / "up")
        assert not (tmp_path / "evil.txt").exists()
