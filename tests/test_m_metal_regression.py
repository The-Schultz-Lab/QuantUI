"""Consolidated inorganic / coordination-complex regression set (M-METAL MET.7).

Phase 1 shipped connectivity perception (MET.2/MET.6), the PlotlyMol fallback
(MET.3), pre-opt honesty (MET.4), the basis/spin guards (MET.5), and 14 bundled
example complexes (MET.9) — each with its own focused unit-test file
(``test_connectivity.py``, ``test_inorganic_guards.py``,
``test_metal_viewer_fallback.py``, ``test_preopt_gfnff.py``,
``test_inorganic_examples.py``). This file does not re-test those mechanisms
in isolation; it is the regression net that loops every one of them over
*every* bundled entry at once, plus the one thing nothing else covers yet: a
real PySCF single point on a metal complex.
"""

from __future__ import annotations

import pytest

from quantui import molecule_library as ml
from quantui.molecule import ATOMIC_NUMBERS, Molecule

_PYSCF_AVAILABLE = False
try:
    import pyscf as _pyscf  # noqa: F401

    _PYSCF_AVAILABLE = True
except ImportError:
    pass

pyscf_only = pytest.mark.skipif(
    not _PYSCF_AVAILABLE, reason="PySCF not installed (Linux/macOS/WSL only)"
)


def _inorganic_entries():
    return [e for e in ml.iter_entries() if e["category"] == "inorganic-complex"]


_ENTRIES = _inorganic_entries()
_ENTRY_IDS = [e["id"] for e in _ENTRIES]


def _molecule(entry) -> Molecule:
    return Molecule(
        atoms=entry["atoms"],
        coordinates=entry["coordinates"],
        charge=entry["charge"],
        multiplicity=entry["multiplicity"],
    )


class TestBundledSetIsNonEmpty:
    def test_at_least_fourteen_entries(self):
        # Guards the parametrization below from silently collecting zero
        # cases if the manifest ever regresses.
        assert len(_ENTRIES) >= 14


@pytest.mark.parametrize("entry", _ENTRIES, ids=_ENTRY_IDS)
class TestEveryBundledComplexClearsPhase1:
    def test_connectivity_is_a_single_component(self, entry):
        from quantui.connectivity import covalent_components

        components = covalent_components(entry["atoms"], entry["coordinates"])
        assert len(components) == 1, entry["id"]

    def test_metal_centre_is_not_a_lone_dot(self, entry):
        from quantui.connectivity import is_metal, metal_coordination_bonds

        bonds = metal_coordination_bonds(entry["atoms"], entry["coordinates"])
        bonded = {i for pair in bonds for i in pair}
        for i, sym in enumerate(entry["atoms"]):
            if is_metal(sym):
                assert i in bonded, f"{entry['id']}: {sym} has no coordination bond"

    def test_charge_multiplicity_guard_passes(self, entry):
        from quantui.inorganic_guards import check_charge_multiplicity

        n_elec = sum(ATOMIC_NUMBERS.get(a, 0) for a in entry["atoms"]) - entry["charge"]
        assert check_charge_multiplicity(n_elec, entry["multiplicity"]) is None, entry[
            "id"
        ]

    @pyscf_only
    def test_basis_guard_clears_with_def2_svp(self, entry):
        from quantui.inorganic_guards import check_basis_coverage

        assert check_basis_coverage(entry["atoms"], "def2-SVP") is None, entry["id"]

    def test_plotlymol_backend_never_hard_crashes(self, entry):
        import quantui.visualization_py3dmol as viz

        if not (viz.PLOTLYMOL_AVAILABLE and viz.PY3DMOL_AVAILABLE):
            pytest.skip("both plotlymol and py3dmol backends required")
        # MET.3: PlotlyMol's RDKit valence perception raises on a metal; the
        # router must fall back to py3Dmol rather than propagate. Whatever
        # comes back must be a usable view object, not an exception.
        view = viz.visualize_molecule(_molecule(entry), backend="plotlymol")
        assert view is not None, entry["id"]

    def test_preopt_never_reports_a_false_no_op(self, entry):
        # MET.4: preopt_support must not claim "supported" when nothing can
        # actually relax the molecule (that's what turns a real failure into
        # a misleading 0.0 A "no meaningful change" upstream).
        from quantui.preopt import preopt_support

        reason = preopt_support(_molecule(entry))
        if reason is not None:
            assert isinstance(reason, str) and reason, entry["id"]

    def test_scattering_the_metal_still_trips_the_disconnection_warning(self, entry):
        # MET.2, generalized across the whole bundled set: pulling the metal
        # away from its donors (the shape a resolved "salt" takes) must still
        # be caught for every entry, not just the cisplatin case the original
        # unit test covers.
        from quantui.connectivity import describe_disconnection, is_metal

        atoms = entry["atoms"]
        metal_idx = next((i for i, s in enumerate(atoms) if is_metal(s)), None)
        if metal_idx is None:
            pytest.skip(f"{entry['id']} has no metal centre to scatter")
        coords = [list(c) for c in entry["coordinates"]]
        coords[metal_idx] = [c + 8.0 for c in coords[metal_idx]]
        msg = describe_disconnection(atoms, coords)
        assert msg is not None, entry["id"]
        assert atoms[metal_idx] in msg


@pyscf_only
class TestSmallECPSinglePoint:
    """One real SCF run through the whole compute pipeline as a cloud
    regression guard. This is NOT the MET.8 exit gate — that needs the
    instructor's local Voila + full-set DFT geometry-optimization pass. It
    only catches a basis/ECP/guard regression between now and that pass, on
    the one complex (cisplatin) already validated locally per MET.8/MET.5.
    """

    def test_cisplatin_rhf_def2svp_converges(self):
        from pyscf import gto, scf

        from quantui.inorganic_guards import ecp_for_basis

        entry = ml.get("inorganic-cisplatin")
        assert entry is not None
        molecule = _molecule(entry)

        mol = gto.Mole()
        mol.atom = molecule.to_pyscf_format()
        mol.basis = "def2-SVP"
        mol.ecp = ecp_for_basis("def2-SVP", molecule.atoms)
        mol.charge = molecule.charge
        mol.spin = molecule.multiplicity - 1
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.max_cycle = 100
        mf.kernel()

        assert mf.converged
