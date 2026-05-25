"""Regression tests for the four bugs reported in session 55 (2026-05-25).

Bug A — GPU-run results saved with no MO data
    ``_run_session_calc_body`` extracts ``mf.mo_energy`` / ``mo_coeff`` /
    ``mo_occ`` via ``numpy.array(...)``. With a GPU-offloaded ``mf`` those
    are CuPy arrays — numpy refuses implicit device transfers, so the
    bare ``except`` swallowed a ``TypeError`` and the SessionResult
    shipped with all MO fields ``None``. That made ``save_orbitals``
    no-op and history replay of any GPU-run SP/GeoOpt rendered "Not
    available" in Energies + Isosurface panels.

Bug B1/B2/B3 — Calculate-tab molecule viewer used the
    ``with self.viz_output: display_molecule(...)`` pattern. Symptoms:
    initial render wouldn't appear after a PubChem search (B1);
    PlotlyMol RDKit valence errors spilled out as red logger lines
    around the viewer (B2); generic ``logger.info`` lines from the
    renderer were captured into the Output widget (B3). Fix migrates
    to ``_refresh_calc_mol_viewer`` which renders HTML outside any
    Output context and atomic-swaps into ``viz_output``.

Bug C — Frequency pre-opt on benzene crashed the whole calc with
    "singular matrix" in PySCF's ``cho_solve``. Three pre-opt sites
    in ``_do_run`` now ``try/except`` around ``optimize_geometry`` and
    fall back to the user-provided geometry on failure.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

# =====================================================================
# Bug A — cupy-aware MO array extraction in session_calc
# =====================================================================


class _FakeCupyArray:
    """A minimal stand-in for a CuPy array: numpy refuses to convert it
    directly, but it exposes ``.get()`` (sync device→host copy) and
    its ``type(...).__module__`` starts with ``"cupy"`` — the two
    properties the fix probes."""

    def __init__(self, host_data):
        self._host = np.asarray(host_data)

    def get(self):
        return self._host

    # numpy.asarray on a non-array-like falls back to object dtype unless
    # we make the conversion explicitly fail like the real cupy.
    def __array__(self, dtype=None):
        raise TypeError(
            "Implicit conversion to a NumPy array is not allowed. "
            "Please use `.get()` to construct a NumPy array explicitly."
        )


# Pin __module__ so the type probe matches.
_FakeCupyArray.__module__ = "cupy._core.core"


def _extract_to_numpy(arr):
    """Re-implementation of the closure to keep the test independent of
    session_calc's import side effects. Mirrors the production helper:
    detect CuPy by ``.get()`` callable + module prefix, otherwise pass
    through ``np.asarray``."""
    if arr is None:
        return None
    get = getattr(arr, "get", None)
    if callable(get) and type(arr).__module__.startswith("cupy"):
        return np.asarray(get())
    return np.asarray(arr)


class TestBugA_CupyAwareConversion:
    def test_none_passes_through(self):
        assert _extract_to_numpy(None) is None

    def test_numpy_array_passes_through(self):
        a = np.array([1.0, 2.0, 3.0])
        out = _extract_to_numpy(a)
        np.testing.assert_array_equal(out, a)

    def test_cupy_like_is_converted_via_get(self):
        fake = _FakeCupyArray([4.0, 5.0, 6.0])
        out = _extract_to_numpy(fake)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, [4.0, 5.0, 6.0])

    def test_bare_numpy_conversion_of_cupy_like_raises(self):
        # Sanity: the production fix is needed precisely because the
        # naive call (pre-fix code) raises. If this test ever stops
        # raising, the regression guard is moot.
        fake = _FakeCupyArray([1.0])
        with pytest.raises(TypeError):
            np.array(fake)

    def test_production_helper_uses_to_numpy_array(self):
        # Confirm the actual session_calc body contains the
        # ``_to_numpy_array`` helper (so a future refactor that drops it
        # breaks this test loudly).
        from quantui import session_calc

        src = inspect.getsource(session_calc)
        assert "_to_numpy_array" in src
        assert "cupy" in src.lower()


# =====================================================================
# Bug B — Calculate-tab molecule viewer uses atomic HTML swap
# =====================================================================


class TestBugB_AtomicMolViewerSwap:
    def test_app_has_refresh_calc_mol_viewer(self):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        assert hasattr(app, "_refresh_calc_mol_viewer")

    def test_refresh_calc_mol_viewer_handles_none_molecule(self):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        # No molecule loaded yet → must return cleanly, not raise.
        assert app._molecule is None
        app._refresh_calc_mol_viewer()  # should not raise

    def test_calc_tab_does_not_use_with_viz_output_display_pattern(self):
        # The BUG.7 pattern (Analysis tab) and this bug-batch's fix both
        # forbid the ``with self.viz_output: display_molecule(...)``
        # idiom. Verify no occurrence remains in the migrated section.
        from quantui import app as _app_mod

        src = inspect.getsource(_app_mod)
        # ``_display_molecule`` is the imported alias; the fix removed
        # all 5 of its call sites. The module may still import it for
        # backwards compat, so we only check that the buggy
        # idiom (``with self.viz_output:`` followed by a
        # ``_display_molecule`` call) is gone.
        idx = 0
        while True:
            idx = src.find("with self.viz_output:", idx)
            if idx < 0:
                break
            # Look at the next ~200 characters for a _display_molecule
            # call. If we find one, the bad idiom is still present.
            window = src[idx : idx + 400]
            assert "_display_molecule(" not in window, (
                "Found ``with self.viz_output: _display_molecule(...)`` "
                "idiom; should be migrated to _refresh_calc_mol_viewer "
                "(BUG B1/B2/B3)."
            )
            idx += 1


# =====================================================================
# Bug C — Pre-opt failures fall back to user geometry instead of crashing
# =====================================================================


class TestBugC_PreoptFailureFallback:
    def test_freq_preopt_block_has_try_except(self):
        # Confirm the source contains the new fallback paths. Reading
        # the source is the most direct way to assert this; running the
        # actual freq calc would require PySCF.
        #
        # POLISH.9 (2026-05-25) renamed user-facing "Pre-optimisation"
        # → "Geometry optimization"; update the guard string to match.
        from quantui import app as _app_mod

        src = inspect.getsource(_app_mod)
        assert "Geometry optimization failed" in src
        # The exception variable name (_pre_exc) is unique to the new
        # try/except wrapping all three pre-opt sites.
        assert src.count("except Exception as _pre_exc") >= 3

    def test_freq_preopt_fallback_uses_user_geometry(self):
        # The fallback message should make it clear the calc continues
        # with the user-provided geometry — that's the contract the bug
        # report asked for.
        from quantui import app as _app_mod

        src = inspect.getsource(_app_mod)
        assert "user-provided geometry" in src or "seed geometry as-is" in src
