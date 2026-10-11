"""
PySCF's input dump must not copy the running script into job logs
(GOTCHAS 2026-10-10: every batch live.log carried ~900 lines of worker.py).
"""

from __future__ import annotations

import io

import pytest

pytest.importorskip("pyscf")

import __main__  # noqa: E402
from quantui.session_calc import build_mole  # noqa: E402


def _mol(stream):
    from pyscf import gto

    mol = gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 0.74"
    mol.basis = "sto-3g"
    mol.verbose = 4
    mol.stdout = stream
    return mol


def test_script_is_left_out_but_the_input_summary_stays(tmp_path, monkeypatch):
    script = tmp_path / "worker.py"
    script.write_text("SECRET_SOURCE_LINE = 1\n")
    monkeypatch.setattr(__main__, "__file__", str(script), raising=False)
    out = io.StringIO()
    build_mole(_mol(out))
    text = out.getvalue()
    assert "SECRET_SOURCE_LINE" not in text
    assert "input file is" not in text
    assert "[INPUT] num. atoms = 2" in text
    assert __main__.__file__ == str(script)  # restored


def test_without_a_main_file_it_is_a_plain_build(monkeypatch):
    monkeypatch.delattr(__main__, "__file__", raising=False)
    out = io.StringIO()
    build_mole(_mol(out))
    assert "[INPUT] num. atoms = 2" in out.getvalue()
    assert not hasattr(__main__, "__file__")
