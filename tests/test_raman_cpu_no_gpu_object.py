"""CPU Raman FD loop never hands a gpu4pyscf object to CPU
pyscf-properties (audit SCF-9).

Each displaced SCF went through ``try_to_gpu`` and was then passed to
``pyscf.prop.polarizability`` (NumPy code). On a GPU host (every
open-shell frequency job takes this path) that is a GPU object in CPU
code. The loop now stays on CPU and brings a CuPy reference density to the
host first. No GPU here, so the GPU side is mocked.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

pytest.importorskip("pyscf")
pytest.importorskip("pyscf.prop.polarizability")


class _FakeCupyArray:
    """Looks like a CuPy array to the host-copy probe; NumPy refuses it."""

    __module__ = "cupy._core.core"

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def get(self):
        return self._arr.copy()

    def __array__(self, *a, **k):
        raise TypeError("Implicit conversion to a NumPy array is not allowed")


def _h2_inputs():
    from pyscf import dft, gto, scf

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    nm = np.array([[[0.0, 0.0, -0.7071], [0.0, 0.0, 0.7071]]])
    return (
        dict(
            mf=mf,
            mol=mol,
            scf=scf,
            dft=dft,
            displacements=nm.tolist(),
            frequencies_cm1=[5000.0],
            dm0_is_unrestricted=False,
            density_fit_used=False,
            stream=io.StringIO(),
            status=lambda _m: None,
        ),
        mf.make_rdm1(),
    )


def test_displaced_scfs_are_not_offloaded(monkeypatch):
    import quantui.gpu_offload as go
    from quantui.raman_calc import _cpu_raman_activities_fd

    monkeypatch.setenv("QUANTUI_FREQ_PARALLEL", "0")
    offloaded: list = []

    def fake_to_gpu(mf, method_upper):
        offloaded.append(type(mf).__name__)
        return mf, True, "FakeGPU"

    monkeypatch.setattr(go, "try_to_gpu", fake_to_gpu)
    kwargs, dm0 = _h2_inputs()
    out = _cpu_raman_activities_fd(dm0=dm0, **kwargs)
    assert offloaded == []
    assert len(out) == 1 and out[0] > 0


def test_cupy_reference_density_is_brought_to_host(monkeypatch):
    from quantui.raman_calc import _cpu_raman_activities_fd

    monkeypatch.setenv("QUANTUI_FREQ_PARALLEL", "0")
    kwargs, dm0 = _h2_inputs()
    ref = _cpu_raman_activities_fd(dm0=dm0, **kwargs)
    out = _cpu_raman_activities_fd(dm0=_FakeCupyArray(dm0), **kwargs)
    assert out == pytest.approx(ref, rel=1e-6)
