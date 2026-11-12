"""IRC fallback after Sella's inner-loop failure: the stored end is the minimum.

Sella can raise ``IRCInnerLoopConvergenceFailure`` after moving the atoms to
an inner-iteration geometry no observer recorded. The branch is then finished
by minimizing from where the atoms are; its last frame (end structure and
energy, which set the barriers) must be that minimized geometry, not the last
IRC point. Fast: a harmonic stand-in replaces PySCF and Sella's IRC.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from quantui.molecule import Molecule

ATOMS = ["C", "N", "H"]
START = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.18], [1.0, 0.0, 0.6]]
# The minimum the stand-in potential has on the "forward" side.
MINIMUM = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.18], [1.0, 0.0, -0.4]])


def _harmonic_calc_class():
    from ase.calculators.calculator import Calculator, all_changes

    class Harmonic(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, **_kwargs):
            super().__init__()

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            d = self.atoms.get_positions() - MINIMUM
            self.results = {"energy": float(0.5 * (d**2).sum()), "forces": -d}

    return Harmonic


def _fake_sella_irc(end_positions):
    from sella.optimize.irc import IRCInnerLoopConvergenceFailure

    class FakeIRC:
        """Records the start, moves to *end_positions* unrecorded, then fails."""

        def __init__(self, atoms, **_kwargs):
            self.atoms = atoms
            self.observers = []
            self.nsteps = 0

        def attach(self, fn, interval=1):
            self.observers.append(fn)

        def run(self, fmax, steps, direction):
            self.atoms.set_positions(np.array(START))
            for fn in self.observers:
                fn()
            self.atoms.set_positions(np.array(end_positions))
            raise IRCInnerLoopConvergenceFailure

    return FakeIRC


def _run(monkeypatch, end_positions):
    pytest.importorskip("sella")
    import sella

    import quantui.optimizer as optimizer
    import quantui.ts_search as ts_search
    from quantui.irc import run_irc

    monkeypatch.setattr(optimizer, "_QuantUIPySCFCalc", _harmonic_calc_class())
    monkeypatch.setattr(ts_search, "_analytic_hessian_fn", lambda calc: None)
    monkeypatch.setattr(sella, "IRC", _fake_sella_irc(end_positions))
    return run_irc(
        Molecule(ATOMS, START, validate_spin=False),
        "RHF",
        "STO-3G",
        fmax=0.05,
        progress_stream=io.StringIO(),
    )


def _end_force(branch) -> float:
    d = np.asarray(branch.trajectory[-1].coordinates) - MINIMUM
    return float(np.sqrt((d**2).sum(axis=1)).max())


def test_failure_at_the_minimum_stores_the_minimum(monkeypatch):
    # Sella's failed inner loop already sits at the minimum: the descent
    # takes no step. The branch used to end on the previous IRC point (here
    # the start) while reporting "reached a minimum".
    r = _run(monkeypatch, MINIMUM)
    for branch in (r.forward, r.reverse):
        assert branch.converged
        assert _end_force(branch) < 0.05
        assert branch.end_energy_hartree == pytest.approx(0.0, abs=1e-12)


def test_failure_near_the_minimum_ends_converged_without_duplicates(monkeypatch):
    near = MINIMUM + np.array([[0, 0, 0], [0, 0, 0], [0.3, 0.0, 0.2]])
    r = _run(monkeypatch, near)
    for branch in (r.forward, r.reverse):
        assert branch.converged
        assert _end_force(branch) < 0.05
        xyz = [np.asarray(f.coordinates) for f in branch.trajectory]
        steps = [np.abs(b - a).max() for a, b in zip(xyz, xyz[1:])]
        assert min(steps) > 0, "a frame is repeated"
