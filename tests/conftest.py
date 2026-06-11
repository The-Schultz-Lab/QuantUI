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
