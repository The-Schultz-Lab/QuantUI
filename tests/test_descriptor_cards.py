"""Tests for the UXP.7 method / basis descriptor-card builders.

Pure string builders — no widgets, no PySCF — so they exercise the family
classification, the icon embedding, and the one-line copy directly.
"""

from __future__ import annotations

import pytest

from quantui import config
from quantui.descriptor_cards import (
    basis_card_html,
    basis_family,
    method_card_html,
)


class TestBasisFamily:
    @pytest.mark.parametrize(
        "basis,expected",
        [
            ("STO-3G", "minimal"),
            ("3-21G", "pople"),
            ("6-31G", "pople"),
            ("6-31G*", "pople"),
            ("6-31G**", "pople"),
            ("cc-pVDZ", "cc"),
            ("cc-pVTZ", "cc"),
            ("def2-SVP", "def2"),
            ("def2-TZVP", "def2"),
        ],
    )
    def test_family_classification(self, basis, expected):
        assert basis_family(basis) == expected

    def test_every_supported_basis_has_a_family_and_card(self):
        # No supported basis should fall through to a KeyError in the copy map.
        for basis in config.SUPPORTED_BASIS_SETS:
            html = basis_card_html(basis)
            assert basis in html
            assert "<svg" in html


class TestMethodCard:
    def test_every_supported_method_renders(self):
        for method in config.SUPPORTED_METHODS:
            html = method_card_html(method)
            assert method in html
            assert "<svg" in html
            # One-line body comes from METHOD_INFO.use_for.
            assert config.METHOD_INFO[method]["use_for"] in html

    def test_family_accent_colour_differs_by_type(self):
        # hf / dft / wavefunction should not all share one accent colour.
        hf = method_card_html("RHF")
        dft = method_card_html("B3LYP")
        wfn = method_card_html("CCSD")
        assert "var(--q-accent-info)" in hf
        assert "var(--q-accent-purple)" in dft
        assert "var(--q-accent-warning)" in wfn

    def test_unknown_method_does_not_raise(self):
        html = method_card_html("NOT_A_METHOD")
        assert "NOT_A_METHOD" in html
        assert "<svg" in html

    def test_title_includes_family_descriptor(self):
        # "B3LYP · DFT Hybrid Functional" — descriptor pulled from the label.
        html = method_card_html("B3LYP")
        assert "B3LYP ·" in html


class TestCardStructure:
    def test_card_uses_left_border_accent_like_result_cards(self):
        html = method_card_html("RHF")
        assert "border-left:4px solid" in html

    def test_no_cdn_or_external_asset(self):
        # Offline-safe: inline SVG only, never an <img src=http...> / CDN link.
        for html in (method_card_html("PBE"), basis_card_html("cc-pVTZ")):
            assert "http://" not in html
            assert "https://" not in html
