"""
M-BATCH2 B2.4 — resource estimator: def2/LANL2DZ factors, Pople aliases,
site-configurable caps, walltime for TD-DFT states and PES scan points.
"""

from __future__ import annotations

import pytest

from quantui.backends import cluster_config as cfg
from quantui.backends.base import CalculationRequest
from quantui.backends.slurm_utils import basis_factor, estimate_slurm_resources


@pytest.mark.parametrize(
    "basis,factor",
    [
        ("def2-SVP", 3.0),
        ("def2-svp", 3.0),
        ("def2-TZVP", 4.5),
        ("LANL2DZ", 1.5),
        ("6-31G(d)", 2.0),
        ("6-31G*", 2.0),
        ("6-31G(d,p)", 2.5),
        ("6-31G**", 2.5),
        ("cc-pvdz", 3.0),
        ("some-new-basis", 2.0),
    ],
)
def test_basis_factor(basis, factor):
    assert basis_factor(basis) == factor


def test_every_supported_basis_has_a_factor():
    from quantui.backends.slurm_utils import BASIS_FACTORS
    from quantui.config import SUPPORTED_BASIS_SETS

    assert set(SUPPORTED_BASIS_SETS) <= set(BASIS_FACTORS)


def _req(calc="single_point", atoms=20, options=None, basis="def2-SVP"):
    return CalculationRequest(
        request_id="r",
        calc_type=calc,
        method="B3LYP",
        basis=basis,
        charge=0,
        multiplicity=1,
        molecule={"atoms": ["C"] * atoms, "coords": [[i, 0, 0] for i in range(atoms)]},
        options=options or {},
    )


def test_estimate_ceiling_is_separate_from_the_hard_limit(monkeypatch):
    big = _req("frequency", atoms=40)
    monkeypatch.setattr(cfg, "MAX_MEMORY_GB", 512)
    monkeypatch.setattr(cfg, "ESTIMATE_MAX_MEMORY_GB", 128)
    assert estimate_slurm_resources(big)["memory_gb"] == 128
    monkeypatch.setattr(cfg, "ESTIMATE_MAX_MEMORY_GB", 512)
    assert estimate_slurm_resources(big)["memory_gb"] > 128


def test_caps_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("QUANTUI_SLURM_MAX_MEMORY_GB", "512")
    monkeypatch.setenv("QUANTUI_SLURM_ESTIMATE_MAX_MEMORY_GB", "900")
    monkeypatch.setenv("QUANTUI_SLURM_MAX_CORES", "bad")
    import importlib

    try:
        mod = importlib.reload(cfg)
        assert mod.MAX_MEMORY_GB == 512
        assert mod.ESTIMATE_MAX_MEMORY_GB == 512  # never above the hard limit
        assert mod.MAX_CORES == 32  # invalid value -> default
    finally:
        monkeypatch.delenv("QUANTUI_SLURM_MAX_MEMORY_GB")
        monkeypatch.delenv("QUANTUI_SLURM_ESTIMATE_MAX_MEMORY_GB")
        monkeypatch.delenv("QUANTUI_SLURM_MAX_CORES")
        importlib.reload(cfg)


@pytest.mark.parametrize(
    "calc,key,values",
    [("tddft", "nstates", (10, 20, 40)), ("pes_scan", "steps", (10, 20, 40))],
)
def test_walltime_grows_with_states_and_scan_points(calc, key, values):
    times = [
        estimate_slurm_resources(_req(calc, atoms=3, options={key: v}))["walltime"]
        for v in values
    ]
    idx = [cfg.WALLTIME_OPTIONS.index(t) for t in times]
    assert idx[0] < idx[1] < idx[2]
