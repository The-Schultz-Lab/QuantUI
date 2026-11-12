"""Regression tests for the Low-priority visualization audit items (VIZ-7..14)."""

from __future__ import annotations

import numpy as np
import pytest

# ── VIZ-7: generate_cube_file must not unpickle results.npz ────────────────


def _h2_npz(path, *, mol_atom):
    np.savez(
        path,
        mo_coeff=np.array([[0.55, 1.2], [0.55, -1.2]]),
        mo_occ=np.array([2.0, 0.0]),
        mol_atom=mol_atom,
        mol_basis="STO-3G",
    )


class TestGenerateCubeFileNoPickle:
    def test_object_array_in_npz_is_refused_not_unpickled(self, tmp_path):
        pytest.importorskip("pyscf")
        from quantui.orbital_visualization import generate_cube_file

        npz = tmp_path / "results.npz"
        # A list of (symbol, coords) tuples is saved as a pickled object array.
        _h2_npz(
            npz,
            mol_atom=np.array([("H", (0, 0, 0)), ("H", (0, 0, 0.74))], dtype=object),
        )
        with pytest.raises(ValueError, match="allow_pickle"):
            generate_cube_file(npz, 0, tmp_path / "mo.cube", nx=4, ny=4, nz=4)

    def test_plain_string_npz_still_works(self, tmp_path):
        pytest.importorskip("pyscf")
        from quantui.orbital_visualization import generate_cube_file

        npz = tmp_path / "results.npz"
        _h2_npz(npz, mol_atom="H 0 0 0; H 0 0 0.74")
        out = generate_cube_file(npz, 0, tmp_path / "mo.cube", nx=4, ny=4, nz=4)
        assert out.exists()
