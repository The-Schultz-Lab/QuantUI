"""Opt-in real NVIDIA tests for geometry-optimization offload.

Normal CI does not provide a CUDA device. Set
``QUANTUI_RUN_GPU_INTEGRATION=1`` on a Linux/WSL host with gpu4pyscf and a
working NVIDIA driver to execute the real gate.
"""

from __future__ import annotations

import os

import pytest

from quantui.molecule import Molecule


@pytest.mark.gpu_integration
@pytest.mark.skipif(
    os.environ.get("QUANTUI_RUN_GPU_INTEGRATION") != "1",
    reason="set QUANTUI_RUN_GPU_INTEGRATION=1 for the real NVIDIA check",
)
def test_geometry_optimizer_offloads_each_pyscf_scf_step():
    from quantui.gpu_offload import probe_gpu
    from quantui.optimizer import optimize_geometry

    available, name, reason = probe_gpu()
    if not available:
        pytest.fail(f"GPU integration requested but unavailable: {reason}")

    result = optimize_geometry(
        Molecule(
            ["H", "H"],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        ),
        method="RHF",
        basis="STO-3G",
        fmax=0.05,
        steps=1,
    )

    assert result.gpu_used is True
    assert result.gpu_name == name
    assert result.converged or result.n_steps == 1


@pytest.mark.pyfock_gpu
@pytest.mark.skipif(
    os.environ.get("QUANTUI_RUN_PYFOCK_GPU_INTEGRATION") != "1",
    reason="set QUANTUI_RUN_PYFOCK_GPU_INTEGRATION=1 for the real PyFock GPU check",
)
def test_pyfock_gpu_water_single_point_is_real():
    from quantui.engines import EngineRequest, PyfockEngine

    result = PyfockEngine().run(
        EngineRequest(
            request_id="pyfock-gpu-water",
            calc_type="single_point",
            method="PBE",
            basis="def2-SVP",
            charge=0,
            multiplicity=1,
            molecule={
                "atoms": ["O", "H", "H"],
                "coordinates": [
                    [0.0, 0.0, 0.11779],
                    [0.0, 0.755453, -0.471161],
                    [0.0, -0.755453, -0.471161],
                ],
            },
            options={"use_gpu": True, "ncores": 2, "max_iterations": 50},
        )
    )
    assert result.status == "success", result.error
    assert result.converged is True
    assert result.native_result.gpu_used is True
    assert result.native_result.gpu_name
