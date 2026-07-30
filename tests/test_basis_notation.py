"""Tests for basis-set notation clarity (M-UX2 UXP2.1).

``6-31G*`` and ``6-31G(d)`` are the same basis set written two ways. Nothing in
the UI used to say so, which leaves a student who learned one notation assuming
the other is a different, unavailable option. These tests lock in that the
equivalence is stated in the descriptor card, the educational notes, and the
basis-set help topic.

Platform-independent: no PySCF required. (The claim that PySCF accepts *both*
spellings was verified separately against real basis construction — identical AO
counts for each pair — and is asserted here only as documented copy.)
"""

from __future__ import annotations

import pytest

from quantui import config, descriptor_cards
from quantui.help_content import HELP_TOPICS


class TestPopleNotationAlias:
    @pytest.mark.parametrize(
        ("basis", "expected"),
        [
            ("6-31G*", "6-31G(d)"),
            ("6-31G**", "6-31G(d,p)"),
            # Not in the dropdown today, but the mapping is derived so a future
            # dropdown entry gets the alias for free.
            ("6-311G**", "6-311G(d,p)"),
            ("6-31+G*", "6-31+G(d)"),
            ("6-311++G**", "6-311++G(d,p)"),
        ],
    )
    def test_starred_names_get_parenthesised_alias(self, basis, expected):
        assert config.pople_notation_alias(basis) == expected

    @pytest.mark.parametrize(
        "basis",
        ["6-31G", "STO-3G", "3-21G", "cc-pVDZ", "cc-pVTZ", "def2-SVP", "def2-TZVP"],
    )
    def test_unstarred_names_have_no_alias(self, basis):
        # These have no alternate spelling; inventing one would be wrong.
        assert config.pople_notation_alias(basis) == ""

    def test_double_star_checked_before_single(self):
        # Order matters: a "*"-first check would turn 6-31G** into "6-31G*(d)".
        assert config.pople_notation_alias("6-31G**") == "6-31G(d,p)"


class TestBasisCardShowsAlias:
    def test_starred_basis_card_states_equivalence(self):
        html = descriptor_cards.basis_card_html("6-31G*")
        assert "6-31G(d)" in html
        assert "Also written" in html

    def test_double_starred_basis_card(self):
        html = descriptor_cards.basis_card_html("6-31G**")
        assert "6-31G(d,p)" in html

    def test_unstarred_basis_card_has_no_alias_line(self):
        html = descriptor_cards.basis_card_html("cc-pVDZ")
        assert "Also written" not in html

    def test_alias_line_stays_short(self):
        # The cards exist because the old inline notes were "a lot of word
        # clutter" (FR-DESCRIPTOR-CARDS). Guard against the alias growing into a
        # paragraph: the starred card should not be dramatically longer than the
        # unstarred one.
        starred = descriptor_cards.basis_card_html("6-31G*")
        plain = descriptor_cards.basis_card_html("6-31G")
        assert len(starred) - len(plain) < 200

    def test_every_supported_basis_renders(self):
        # Whatever is in the dropdown must produce a card without raising.
        for basis in config.SUPPORTED_BASIS_SETS:
            assert descriptor_cards.basis_card_html(basis)


class TestEducationalNotesStateEquivalence:
    def _notes(self, basis: str) -> str:
        from quantui.calculator import PySCFCalculation
        from quantui.molecule import Molecule

        mol = Molecule(atoms=["H", "H"], coordinates=[[0, 0, 0], [0, 0, 0.74]])
        return PySCFCalculation(mol, method="RHF", basis=basis).get_educational_notes()

    def test_starred_basis_notes_mention_alias(self):
        assert "6-31G(d)" in self._notes("6-31G*")

    def test_unstarred_basis_notes_do_not(self):
        assert "(d)" not in self._notes("6-31G")


class TestHelpTopicExplainsNotation:
    @property
    def body(self) -> str:
        return HELP_TOPICS["basis_set"]["body"]

    def test_both_notations_documented(self):
        for token in ("6-31G*", "6-31G(d)", "6-31G**", "6-31G(d,p)"):
            assert token in self.body

    def test_says_they_are_the_same_set(self):
        assert "same basis set" in self.body

    def test_diffuse_functions_explained(self):
        assert "diffuse" in self.body
        assert "anion" in self.body.lower()

    def test_dunning_contrast_explained(self):
        # The absence of a star on cc-pVDZ is itself a confusion source.
        assert "aug-cc-pVDZ" in self.body
        assert "cc-pVDZ" in self.body
