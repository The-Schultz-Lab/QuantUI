"""Diffuse basis sets in the basis menu (6-31+G*, 6-31++G**, aug-cc-pVDZ, def2-SVPD).

Added for anion / radical-anion work, where a basis without diffuse functions
gives a badly described extra electron. These tests pin that every consumer of
``config.SUPPORTED_BASIS_SETS`` accepts the new names; the PySCF-gated ones
check the names resolve and a UKS anion actually converges.
"""

from __future__ import annotations

import pytest

from quantui import config

DIFFUSE = ("6-31+G*", "6-31++G**", "aug-cc-pVDZ", "def2-SVPD")
PARENTS = ("6-31G*", "6-31G**", "cc-pVDZ", "def2-SVP")


class TestBasisList:
    def test_diffuse_sets_in_menu(self):
        for basis in DIFFUSE:
            assert basis in config.SUPPORTED_BASIS_SETS

    def test_diffuse_subset_matches_menu(self):
        assert config.DIFFUSE_BASIS_SETS == frozenset(DIFFUSE)
        assert config.DIFFUSE_BASIS_SETS <= set(config.SUPPORTED_BASIS_SETS)

    def test_each_diffuse_set_follows_its_parent(self):
        # Menu order keeps each diffuse set next to the set it extends.
        menu = config.SUPPORTED_BASIS_SETS
        for parent, diffuse in zip(PARENTS, DIFFUSE):
            assert menu.index(diffuse) == menu.index(parent) + 1

    def test_no_duplicates(self):
        assert len(config.SUPPORTED_BASIS_SETS) == len(set(config.SUPPORTED_BASIS_SETS))


class TestNotation:
    @pytest.mark.parametrize(
        "basis, alias",
        [("6-31+G*", "6-31+G(d)"), ("6-31++G**", "6-31++G(d,p)")],
    )
    def test_pople_alias(self, basis, alias):
        assert config.pople_notation_alias(basis) == alias

    @pytest.mark.parametrize("basis", ["aug-cc-pVDZ", "def2-SVPD"])
    def test_non_pople_has_no_alias(self, basis):
        assert config.pople_notation_alias(basis) == ""

    @pytest.mark.parametrize("basis", list(DIFFUSE) + ["6-31+G(d)", "6-31++G(d,p)"])
    def test_is_diffuse(self, basis):
        assert config.is_diffuse_basis(basis)

    @pytest.mark.parametrize("basis", PARENTS + ("STO-3G", "6-31G(d)", "LANL2DZ"))
    def test_parents_are_not_diffuse(self, basis):
        assert not config.is_diffuse_basis(basis)


class TestEngines:
    def test_pyscf_engine_advertises_diffuse_sets(self):
        from quantui.engines import PyscfEngine

        caps = PyscfEngine().capabilities()
        for basis in DIFFUSE:
            assert basis in caps.supported_basis_sets

    def test_pyfock_does_not(self):
        # PyFock is neutral closed-shell def2-SVP/TZVP only; the app filters the
        # menu by this tuple, so diffuse sets must not leak in.
        from quantui.engines import PyfockEngine

        caps = PyfockEngine().capabilities()
        assert not set(DIFFUSE) & set(caps.supported_basis_sets)


class TestCards:
    @pytest.mark.parametrize(
        "basis, family",
        [
            ("6-31+G*", "pople"),
            ("6-31++G**", "pople"),
            ("aug-cc-pVDZ", "cc"),
            ("def2-SVPD", "def2"),
        ],
    )
    def test_family(self, basis, family):
        from quantui.descriptor_cards import basis_family

        assert basis_family(basis) == family

    @pytest.mark.parametrize("basis", DIFFUSE)
    def test_card_flags_diffuse(self, basis):
        from quantui.descriptor_cards import basis_card_html

        assert "Diffuse" in basis_card_html(basis)

    @pytest.mark.parametrize("basis", PARENTS)
    def test_parent_card_does_not(self, basis):
        from quantui.descriptor_cards import basis_card_html

        assert "Diffuse" not in basis_card_html(basis)

    def test_pople_diffuse_card_still_shows_alias(self):
        from quantui.descriptor_cards import basis_card_html

        assert "6-31+G(d)" in basis_card_html("6-31+G*")


class TestEducationalNotes:
    def _notes(self, basis: str, charge: int = 0, mult: int = 1) -> str:
        from quantui.calculator import PySCFCalculation
        from quantui.molecule import Molecule

        mol = Molecule(
            atoms=["O", "H"],
            coordinates=[[0, 0, 0], [0, 0, 0.97]],
            charge=charge,
            multiplicity=mult,
        )
        return PySCFCalculation(
            mol, method="B3LYP", basis=basis
        ).get_educational_notes()

    def test_diffuse_basis_explained(self):
        notes = self._notes("6-31+G*", charge=-1)
        assert "Diffuse functions" in notes
        # The Pople note (and its alias) still applies to the + form.
        assert "6-31+G(d)" in notes

    def test_anion_on_plain_basis_is_nudged(self):
        assert "Anion without diffuse functions" in self._notes("6-31G*", charge=-1)

    def test_neutral_on_plain_basis_is_not_nudged(self):
        notes = self._notes("6-31G*", charge=0, mult=2)
        assert "diffuse" not in notes.lower()

    def test_anion_on_diffuse_basis_is_not_nudged(self):
        assert "Anion without" not in self._notes("aug-cc-pVDZ", charge=-1)


def test_help_topic_names_every_diffuse_set():
    from quantui.help_content import HELP_TOPICS

    body = HELP_TOPICS["basis_set"]["body"]
    for basis in DIFFUSE:
        assert basis in body
    assert "6-31+G(d)" in body


class TestBatchAndEstimates:
    def test_resource_factors_present_and_above_parent(self):
        from quantui.backends.slurm_utils import BASIS_FACTORS

        for parent, diffuse in zip(PARENTS, DIFFUSE):
            assert BASIS_FACTORS[diffuse] > BASIS_FACTORS.get(parent, 2.0)
        # def2-SVP has cc-pVDZ's size; keep def2-SVPD above that even where
        # def2-SVP has no entry of its own yet.
        assert BASIS_FACTORS["def2-SVPD"] > BASIS_FACTORS["cc-pVDZ"]

    def test_launcher_site_constants_carry_factors(self):
        from quantui.batch_launcher import site_constants

        factors = site_constants()["basis_factors"]
        for basis in DIFFUSE:
            assert basis in factors

    @pytest.mark.parametrize(
        "basis, tag",
        [
            ("6-31+G*", "6-31pGd"),
            ("6-31++G**", "6-31ppGdp"),
            ("aug-cc-pVDZ", "aug-cc-pVDZ"),
            ("def2-SVPD", "def2-SVPD"),
        ],
    )
    def test_job_name_basis(self, basis, tag):
        from quantui.backends.slurm_utils import job_name_basis

        assert job_name_basis(basis) == tag

    def test_basis_function_counts(self):
        from quantui.calc_log import count_basis_functions

        # OH-: O + H, values from PySCF 2.14 (spherical).
        assert count_basis_functions(["O", "H"], "6-31+G*") == 18 + 2
        assert count_basis_functions(["O", "H"], "6-31++G**") == 18 + 6
        assert count_basis_functions(["O", "H"], "aug-cc-pVDZ") == 23 + 9
        assert count_basis_functions(["O", "H"], "def2-SVPD") == 23 + 8


# ── PySCF-gated: the names resolve, and a UKS anion converges ───────────────


@pytest.mark.parametrize("basis", DIFFUSE + ("6-31+G(d)", "6-31++G(d,p)"))
def test_pyscf_resolves_name(basis):
    gto = pytest.importorskip("pyscf.gto")

    mol = gto.M(atom="O 0 0 0; H 0 0 0.97", basis=basis, charge=-1, verbose=0)
    assert mol.nao > 0


@pytest.mark.parametrize("basis", DIFFUSE)
def test_coverage_guard_passes_organic_anion(basis):
    pytest.importorskip("pyscf")
    from quantui.inorganic_guards import check_basis_coverage

    assert check_basis_coverage(["C", "H", "O"], basis) is None


@pytest.mark.filterwarnings("ignore:Basis may be available:UserWarning")
def test_coverage_guard_flags_iodide_on_pople_diffuse():
    # 6-31+G* has no iodine; def2-SVPD (ECP) does.
    pytest.importorskip("pyscf")
    from quantui.inorganic_guards import check_basis_coverage

    assert check_basis_coverage(["I"], "6-31+G*") is not None
    assert check_basis_coverage(["I"], "def2-SVPD") is None


@pytest.mark.integration
@pytest.mark.parametrize("basis", DIFFUSE)
def test_uks_radical_anion_single_point(basis):
    """O2•− (charge −1, doublet) B3LYP single point through the engine path."""
    pytest.importorskip("pyscf")
    from quantui.engines import run_calc
    from quantui.engines.base import EngineRequest

    req = EngineRequest(
        request_id="diffuse",
        calc_type="single_point",
        method="B3LYP",
        basis=basis,
        charge=-1,
        multiplicity=2,
        molecule={"atoms": ["O", "O"], "coordinates": [[0, 0, 0], [0, 0, 1.34]]},
        options={"verbose": 0},
    )
    result = run_calc(req)
    assert result.converged
    assert result.native_result.scf_variant == "UKS"
    # Doublet: ideal <S^2> = 0.75; B3LYP O2- is only lightly contaminated.
    assert abs(result.native_result.spin_square - 0.75) < 0.05
