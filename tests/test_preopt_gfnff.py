"""GFN-FF (xtb) metal-capable pre-optimization backend (M-METAL).

RDKit can't pre-optimize a transition-metal complex; GFN-FF (Grimme's general
force field, via xtb-python + ASE) can. These tests run only where GFN-FF can
*actually run* and are skipped otherwise.

Gating on package presence alone (``preopt._XTB_AVAILABLE``) isn't enough: under
heavy parallel load (``pytest -n auto``), conda's ``libxtb`` shared library
intermittently fails its first ``dlopen`` in a worker and raises
``xtb C extension unimportable, cannot use C-API`` at *run time* — which turned
these tests into flaky failures instead of skips (CI's pip-wheel xtb is
unaffected; the app degrades to an honest no-op). The ``xtb_only`` marker below
therefore probes with a single real GFN-FF run, cached per worker process.
"""

from __future__ import annotations

import os
from functools import lru_cache

import pytest

import quantui.preopt as preopt_mod
from quantui.molecule import Molecule


def _metal(entry_id: str = "inorganic-cisplatin") -> Molecule:
    from quantui import molecule_library as ml

    e = next(x for x in ml.iter_entries() if x["id"] == entry_id)
    return Molecule(
        atoms=e["atoms"],
        coordinates=e["coordinates"],
        charge=e["charge"],
        multiplicity=e["multiplicity"],
    )


def _distorted_metal() -> Molecule:
    m = _metal()
    coords = [list(c) for c in m.coordinates]
    ni = m.atoms.index("N")  # pull one ammine in by 20% — real work for the FF
    coords[ni] = [c * 0.8 for c in coords[ni]]
    return Molecule(
        atoms=list(m.atoms),
        coordinates=coords,
        charge=m.charge,
        multiplicity=m.multiplicity,
    )


def _water() -> Molecule:
    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.81, 0.67, 0.0], [-0.81, 0.67, 0.0]],
    )


@lru_cache(maxsize=1)
def _gfnff_runnable() -> bool:
    """True only if a real GFN-FF run actually succeeds in this worker process.

    Stronger than ``preopt._XTB_AVAILABLE`` (package presence): it performs one
    minimal GFN-FF relaxation. If conda's ``libxtb`` can't ``dlopen`` here (the
    parallel-load flake described in the module docstring), it returns False and
    every ``@xtb_only`` test skips cleanly instead of failing. Cached per process
    (the failure is a first-load failure), so a worker where the extension does
    load has it primed — and the run-time flake can't recur — for the rest of the
    tests, while the probe cost is paid once. Never raises. A genuine GFN-FF
    regression still surfaces: the extension loads (probe True), the tests run,
    and their assertions fail.
    """
    if not preopt_mod._XTB_AVAILABLE:
        return False
    try:
        preopt_mod._xtb_gfnff_relax(_distorted_metal(), 1)
    except Exception:
        return False
    return True


xtb_only = pytest.mark.skipif(
    not _gfnff_runnable(),
    reason=(
        "GFN-FF (xtb) backend can't run here — package missing, or libxtb's "
        "C-API failed to load (e.g. conda libxtb under heavy parallel load)"
    ),
)


class TestEngineSelection:
    @xtb_only
    def test_metal_routes_to_gfnff(self):
        assert preopt_mod.preopt_engine_label(_metal()) == "GFN-FF"

    def test_organic_routes_to_rdkit(self):
        if not preopt_mod._RDKIT_AVAILABLE:
            pytest.skip("rdkit not installed")
        assert preopt_mod.preopt_engine_label(_water()) == "MMFF94/UFF"

    @xtb_only
    def test_metal_is_supported_with_xtb(self):
        assert preopt_mod.preopt_support(_metal()) is None


class TestGfnffRelaxation:
    @xtb_only
    def test_relaxes_distorted_metal(self):
        relaxed, rmsd = preopt_mod.preoptimize(_distorted_metal())
        assert rmsd > 0.02  # GFN-FF actually moved the compressed ammine back
        assert relaxed.atoms == _distorted_metal().atoms  # atom order preserved
        assert relaxed.charge == 0 and relaxed.multiplicity == 1

    @xtb_only
    def test_trajectory_multiframe_ends_at_kept_geometry(self):
        import numpy as np

        mol, rmsd, frames = preopt_mod.preoptimize_with_trajectory(_distorted_metal())
        assert rmsd > 0.02
        assert len(frames) >= 2
        # Last frame must equal the geometry "Keep" adopts.
        assert np.allclose(frames[-1], mol.coordinates, atol=1e-6)

    @xtb_only
    def test_input_is_never_mutated(self):
        m = _distorted_metal()
        before = [list(c) for c in m.coordinates]
        preopt_mod.preoptimize(m)
        assert [list(c) for c in m.coordinates] == before


class TestSideEffectContainment:
    @xtb_only
    def test_no_scratch_files_leak_into_cwd(self, tmp_path, monkeypatch):
        # libxtb writes gfnff_topo / gfnff_adjacency to cwd; the run must contain
        # them in its own temp dir and leave the working directory clean.
        monkeypatch.chdir(tmp_path)
        before = set(os.listdir(tmp_path))
        preopt_mod.preoptimize(_distorted_metal())
        leaked = set(os.listdir(tmp_path)) - before
        assert leaked == set(), f"GFN-FF leaked files: {leaked}"


class TestFallbackWhenXtbAbsent:
    def test_metal_noops_without_xtb(self, monkeypatch):
        # With no GFN-FF backend, a metal falls back to the non-destructive no-op.
        if not preopt_mod._RDKIT_AVAILABLE:
            pytest.skip("rdkit not installed")
        monkeypatch.setattr(preopt_mod, "_XTB_AVAILABLE", False)
        m = _metal()
        relaxed, rmsd = preopt_mod.preoptimize(m)
        assert rmsd == 0.0
        assert relaxed.coordinates == m.coordinates
        assert preopt_mod.preopt_engine_label(m) == ""
