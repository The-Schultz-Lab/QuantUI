"""Open-shell + "RHF" is promoted to UHF in TD-DFT, TS search and IRC (audit GEOM-10).

``scf.RHF`` on an open-shell molecule silently becomes ROHF, which PySCF
cannot use for TDHF ("'NoneType' object is not callable") or for the
analytic Hessian the TS search and IRC need (NotImplementedError).
"""

from __future__ import annotations

import io

import pytest

from quantui.molecule import Molecule
from quantui.reorganization_energy import promote_open_shell_hf

OH = Molecule(["O", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]], multiplicity=2)


class _Stop(Exception):
    pass


def _capture_calc_method(monkeypatch):
    import quantui.optimizer as optimizer

    seen = {}

    class _Calc:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            raise _Stop

    monkeypatch.setattr(optimizer, "_QuantUIPySCFCalc", _Calc)
    return seen


@pytest.mark.parametrize(
    ("method", "mult", "expected"),
    [("RHF", 2, "UHF"), ("hf", 3, "UHF"), ("RHF", 1, "RHF"), ("B3LYP", 2, "B3LYP")],
)
def test_promote_open_shell_hf(method, mult, expected):
    stream = io.StringIO()
    assert promote_open_shell_hf(method, mult, stream) == expected
    assert ("open-shell" in stream.getvalue()) == (expected != method)


def test_ts_search_promotes_rhf(monkeypatch):
    pytest.importorskip("sella")
    from quantui.ts_search import run_ts_search

    seen = _capture_calc_method(monkeypatch)
    stream = io.StringIO()
    with pytest.raises(_Stop):
        run_ts_search(OH, "RHF", "STO-3G", progress_stream=stream)
    assert seen["method"] == "UHF"
    assert "running UHF instead" in stream.getvalue()


def test_irc_promotes_rhf(monkeypatch):
    pytest.importorskip("sella")
    from quantui.irc import run_irc

    seen = _capture_calc_method(monkeypatch)
    with pytest.raises(_Stop):
        run_irc(OH, "RHF", "STO-3G", progress_stream=io.StringIO())
    assert seen["method"] == "UHF"


def test_tddft_rhf_open_shell_runs_as_uhf():
    pytest.importorskip("pyscf")
    from quantui.tddft_calc import run_tddft_calc

    stream = io.StringIO()
    r = run_tddft_calc(OH, "RHF", "6-31G", nstates=2, progress_stream=stream)
    assert r.method == "UHF" and r.scf_variant == "UHF"
    assert r.converged and len(r.excitation_energies_ev) == 2
    assert "TD-DFT failed" not in stream.getvalue()
