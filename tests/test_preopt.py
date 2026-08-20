"""Tests for quantui.preopt — bonded force-field (MMFF94/UFF) pre-optimization.

M-PREOPT / PREOPT.1 replaced the geometry-mangling Lennard-Jones potential with
a bonded RDKit force field and a **non-destructive guarantee**: on any failure
(RDKit missing, bond perception fails, no FF parameters) the original geometry
is returned unchanged rather than a distorted one.

These tests are platform-independent (RDKit is a hard dependency; no PySCF).
"""

import copy
import math

import pytest

from quantui.molecule import Molecule
from quantui.preopt import _RDKIT_AVAILABLE

rdkit_only = pytest.mark.skipif(not _RDKIT_AVAILABLE, reason="rdkit not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _water() -> Molecule:
    """Water with a reasonable (near-equilibrium) geometry."""
    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
    )


def _stretched_water() -> Molecule:
    """Water with mildly stretched O–H bonds (~1.05 Å) — distorted but still
    close enough that RDKit perceives the bonds, so the FF can relax it. (A
    *wildly* broken geometry where bonds aren't perceivable correctly no-ops;
    see the non-destructive guarantee.)"""
    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.81, 0.67, 0.0], [-0.81, 0.67, 0.0]],
    )


def _h2() -> Molecule:
    return Molecule(atoms=["H", "H"], coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])


def _charged_radical() -> Molecule:
    """Water cation (doublet) for charge/multiplicity round-trip checks."""
    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        charge=1,
        multiplicity=2,
    )


def _dist(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _oh_distances(mol: Molecule):
    """O–H distances for a water-ordered (O, H, H) molecule."""
    o = mol.coordinates[0]
    return [_dist(o, mol.coordinates[1]), _dist(o, mol.coordinates[2])]


# ============================================================================
# Non-destructive guarantee (the whole point of M-PREOPT)
# ============================================================================


class TestNonDestructive:
    def test_no_backend_returns_original_unchanged(self, monkeypatch):
        """No pre-opt backend (RDKit and xtb both absent) → return the original
        geometry, never raise."""
        import quantui.preopt as preopt_mod

        monkeypatch.setattr(preopt_mod, "_RDKIT_AVAILABLE", False)
        monkeypatch.setattr(preopt_mod, "_XTB_AVAILABLE", False)
        original = _water()
        mol, rmsd = preopt_mod.preoptimize(original)
        assert isinstance(mol, Molecule)
        assert rmsd == 0.0
        assert mol.coordinates == original.coordinates

    @rdkit_only
    def test_good_water_stays_chemically_intact(self):
        """A good geometry must NOT blow apart (the LJ failure mode)."""
        from quantui.preopt import preoptimize

        optimized, rmsd = preoptimize(_water())
        # Both O–H bonds remain real bonds (LJ pushed these toward ~3.4 Å).
        for d in _oh_distances(optimized):
            assert 0.85 <= d <= 1.2, f"O–H bond unphysical after pre-opt: {d:.3f} Å"
        assert rmsd < 1.0

    @rdkit_only
    def test_distorted_geometry_improves(self):
        """A bond-perceivable distortion should relax toward equilibrium."""
        from quantui.preopt import preoptimize

        eq = 0.96  # MMFF equilibrium O–H ≈ 0.96 Å
        distorted = _stretched_water()
        before = max(_oh_distances(distorted))
        optimized, rmsd = preoptimize(distorted)
        after = max(_oh_distances(optimized))
        # The FF actually engaged (didn't fall back to the original)…
        assert rmsd > 0.0
        # …and the bond ended up closer to equilibrium than it started.
        assert abs(after - eq) < abs(before - eq)


# ============================================================================
# Return type and structure
# ============================================================================


class TestPreoptimizeReturnTypes:
    @rdkit_only
    def test_returns_two_tuple(self):
        from quantui.preopt import preoptimize

        result = preoptimize(_water())
        assert isinstance(result, tuple) and len(result) == 2

    @rdkit_only
    def test_first_element_is_molecule(self):
        from quantui.preopt import preoptimize

        mol, _ = preoptimize(_water())
        assert isinstance(mol, Molecule)

    @rdkit_only
    def test_second_element_is_non_negative_float(self):
        from quantui.preopt import preoptimize

        _, rmsd = preoptimize(_water())
        assert isinstance(rmsd, float) and rmsd >= 0.0


# ============================================================================
# Geometry reasonableness
# ============================================================================


class TestPreoptimizeGeometry:
    @rdkit_only
    def test_atom_count_and_symbols_preserved(self):
        from quantui.preopt import preoptimize

        original = _water()
        optimized, _ = preoptimize(original)
        assert len(optimized.atoms) == len(original.atoms)
        assert optimized.atoms == original.atoms

    @rdkit_only
    def test_coordinates_are_3d_floats(self):
        from quantui.preopt import preoptimize

        optimized, _ = preoptimize(_water())
        for coord in optimized.coordinates:
            assert len(coord) == 3
            assert all(isinstance(v, float) for v in coord)


# ============================================================================
# Metadata preservation (holds whether FF runs or falls back to original)
# ============================================================================


class TestPreoptimizeMetadata:
    @rdkit_only
    def test_neutral_charge_and_multiplicity_preserved(self):
        from quantui.preopt import preoptimize

        optimized, _ = preoptimize(_water())
        assert optimized.charge == 0
        assert optimized.multiplicity == 1

    @rdkit_only
    def test_charge_and_multiplicity_preserved_for_radical(self):
        from quantui.preopt import preoptimize

        original = _charged_radical()
        optimized, _ = preoptimize(original)
        assert optimized.charge == original.charge
        assert optimized.multiplicity == original.multiplicity

    @rdkit_only
    def test_triplet_multiplicity_preserved(self):
        from quantui.preopt import preoptimize

        o2 = Molecule(
            atoms=["O", "O"],
            coordinates=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
            multiplicity=3,
        )
        optimized, _ = preoptimize(o2)
        assert optimized.multiplicity == 3


# ============================================================================
# Input immutability
# ============================================================================


class TestPreoptimizeImmutability:
    @rdkit_only
    def test_original_coordinates_unchanged(self):
        from quantui.preopt import preoptimize

        original = _water()
        original_coords = copy.deepcopy(original.coordinates)
        preoptimize(original)
        assert original.coordinates == original_coords

    @rdkit_only
    def test_original_metadata_unchanged(self):
        from quantui.preopt import preoptimize

        original = _charged_radical()
        preoptimize(original)
        assert original.charge == 1
        assert original.multiplicity == 2


# ============================================================================
# Parameter handling
# ============================================================================


class TestPreoptimizeParameters:
    @rdkit_only
    def test_custom_fmax_accepted(self):
        from quantui.preopt import preoptimize

        _, rmsd = preoptimize(_h2(), fmax=0.5)
        assert isinstance(rmsd, float)

    @rdkit_only
    def test_custom_steps_accepted(self):
        from quantui.preopt import preoptimize

        mol, _ = preoptimize(_h2(), steps=10)
        assert isinstance(mol, Molecule)

    @rdkit_only
    def test_single_step_does_not_crash(self):
        from quantui.preopt import preoptimize

        mol, rmsd = preoptimize(_water(), steps=1)
        assert isinstance(mol, Molecule) and rmsd >= 0.0


# ============================================================================
# Public API surface
# ============================================================================


class TestPreoptimizePublicAPI:
    @rdkit_only
    def test_importable_and_callable_from_quantui(self):
        from quantui import preoptimize

        mol, rmsd = preoptimize(_water())
        assert isinstance(mol, Molecule) and isinstance(rmsd, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
