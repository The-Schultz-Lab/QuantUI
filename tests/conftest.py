"""
Pytest Configuration and Fixtures

Shared test fixtures for QuantUI test suite.
"""

import os
import tempfile

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolate_results_dir():
    """Redirect the default results directory to a temp dir for the whole suite.

    ``results_storage._default_results_dir()`` falls back to a **cwd-relative**
    ``Path("results")`` when ``results_dir`` isn't passed (e.g. the calibration
    runner in ``benchmarks.py`` saves each step that way). Pointing
    ``QUANTUI_RESULTS_DIR`` at a temp dir keeps every default save isolated, so
    the suite never writes ``results/`` into the working directory.
    """
    prev = os.environ.get("QUANTUI_RESULTS_DIR")
    with tempfile.TemporaryDirectory(prefix="quantui_test_results_") as tmp:
        os.environ["QUANTUI_RESULTS_DIR"] = tmp
        yield
        if prev is None:
            os.environ.pop("QUANTUI_RESULTS_DIR", None)
        else:
            os.environ["QUANTUI_RESULTS_DIR"] = prev


@pytest.fixture(autouse=True, scope="session")
def _isolate_user_settings():
    """Point ``QUANTUI_SETTINGS_PATH`` at a temp file for the whole suite.

    Without this the suite reads the developer's real ``~/.quantui/settings.json``
    and results depend on how they last left the app. That is not hypothetical:
    on 2026-07-30 a live GPU.8 verification left ``compute.gpu_enabled = false``
    on disk, and two `quantui gpu check` tests began failing — the probe
    short-circuited at the settings gate and never reached the import branch they
    were asserting on. The tests were correct; their environment was not.

    Every setting is in scope, not just the GPU one: ``viz.default_backend`` and
    ``vib_framerate_fps`` equally feed app construction. Individual tests that
    need specific settings still monkeypatch the same variable per-test, which
    takes precedence over this session default.

    Same rationale as ``_isolate_results_dir`` above, and reflections/10 Rule 6.
    """
    prev = os.environ.get("QUANTUI_SETTINGS_PATH")
    with tempfile.TemporaryDirectory(prefix="quantui_test_settings_") as tmp:
        os.environ["QUANTUI_SETTINGS_PATH"] = os.path.join(tmp, "settings.json")
        yield
        if prev is None:
            os.environ.pop("QUANTUI_SETTINGS_PATH", None)
        else:
            os.environ["QUANTUI_SETTINGS_PATH"] = prev


@pytest.fixture(autouse=True, scope="session")
def _isolate_log_dir():
    """Point ``QUANTUI_LOG_DIR`` at a temp dir for the whole suite.

    Without this the suite appends to the developer's real
    ``~/.quantui/logs/perf_log.jsonl`` — the file the runtime estimator
    trains on. That is not a cosmetic leak. Measured 2026-08-05 while
    scoping M-PROGRESS Phase C: 2 773 records were in the log and roughly
    four fifths of them had been written by tests, not by the user. The
    end-to-end ``*_analysis_history`` suites drive ``_do_run`` with a
    *mocked* calculation, so each one recorded an "H2O RHF/6-31G
    frequency" whose ``elapsed_s`` was really pytest-xdist wall time under
    contention — values from 0.34 s to 143 s for identical chemistry.

    That single fact explains the symptom Phase C was created to fix. The
    estimator was not badly modelled; it was trained almost entirely on
    test-harness noise, which is why its median error was near zero (no
    bias) while its spread was enormous. See
    ``quantui.estimator_eval`` for the replay that measures it.

    Isolating the whole suite is the fix at the source. Tests that assert
    on log behaviour still monkeypatch the same variable per-test, which
    takes precedence over this session default.

    Same rationale as ``_isolate_results_dir`` above, and reflections/10
    Rule 6.
    """
    prev = os.environ.get("QUANTUI_LOG_DIR")
    with tempfile.TemporaryDirectory(prefix="quantui_test_logs_") as tmp:
        os.environ["QUANTUI_LOG_DIR"] = tmp
        yield
        if prev is None:
            os.environ.pop("QUANTUI_LOG_DIR", None)
        else:
            os.environ["QUANTUI_LOG_DIR"] = prev


@pytest.fixture(autouse=True, scope="session")
def _suppress_plotly_browser():
    """Prevent plotly from opening browser tabs during tests.

    plotly's default renderer is "browser" in this env, so any display(fig)
    on a plotly Figure would call pio.show() → open a tab.  Setting
    render_on_display=False makes _ipython_display_ fall through to repr().
    """
    try:
        import plotly.io as pio

        orig_rod = pio.renderers.render_on_display
        pio.renderers.render_on_display = False
        yield
        pio.renderers.render_on_display = orig_rod
    except ImportError:
        yield


@pytest.fixture
def sample_water_xyz():
    """Simple water molecule XYZ coordinates."""
    return """O  0.0  0.0  0.0
H  0.757  0.587  0.0
H  -0.757  0.587  0.0"""


@pytest.fixture
def sample_methane_xyz():
    """Simple methane molecule XYZ coordinates."""
    return """C  0.0  0.0  0.0
H  0.63  0.63  0.63
H  -0.63  -0.63  0.63
H  -0.63  0.63  -0.63
H  0.63  -0.63  -0.63"""


@pytest.fixture
def sample_sdf_water():
    """Sample SDF content for water molecule."""
    return """
  Mrv2311 02131511003D

  3  2  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.7570    0.5870    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7570    0.5870    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
M  END
$$$$
"""


@pytest.fixture
def sample_sdf_metal_complex_2d():
    """Synthetic flat-2D SDF for a Pt(NH3)2Cl2-like coordination complex.

    Mirrors what PubChem/CACTUS actually hand back for a real complex (see
    M-METAL MET.2): no bond entries link the metal to its donor atoms, so
    RDKit's ``GetMolFrags`` sees Pt, each NH3, and each Cl as separate
    fragments even though this is one real coordinated molecule. All-zero Z
    coordinates keep RDKit's 2D/3D autodetection landing on 2D, which is what
    triggers a re-embed for a non-metal input (MET.1's failure mode).
    """
    return """
  Test  2D

  9  4  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 Pt  0  0  0  0  0  0  0  0  0  0  0  0
    2.0000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
    2.3000    0.5000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.3000   -0.5000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.0000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
   -2.3000    0.5000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.3000   -0.5000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    2.0000    0.0000 Cl  0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -2.0000    0.0000 Cl  0  0  0  0  0  0  0  0  0  0  0  0
  2  3  1  0  0  0  0
  2  4  1  0  0  0  0
  5  6  1  0  0  0  0
  5  7  1  0  0  0  0
M  END
$$$$
"""


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create a temporary directory for test files."""
    test_dir = tmp_path / "quantui_test"
    test_dir.mkdir()
    return test_dir


# Markers for test categorization
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "network: tests that require network connectivity"
    )
    config.addinivalue_line("markers", "slow: tests that take significant time to run")
    config.addinivalue_line(
        "markers", "integration: integration tests requiring external services"
    )
