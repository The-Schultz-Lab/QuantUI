"""Tests for the M-EXPORT / EXPORT.3 + EXPORT.7 trajectory writers.

Both formats are platform-independent — they don't require PySCF, only
NumPy (for the Molecule constructor) and ASE (already a QuantUI extra).
ASE-side tests round-trip via :class:`ase.io.trajectory.Trajectory` so
we catch any drift between the writer's output and ASE's reader.
"""

from __future__ import annotations

import pytest

from quantui.molecule import Molecule
from quantui.results_storage import save_trajectory_ase, save_trajectory_xyz

_ASE_AVAILABLE = False
try:
    import ase  # noqa: F401

    _ASE_AVAILABLE = True
except ImportError:
    pass

ase_only = pytest.mark.skipif(
    not _ASE_AVAILABLE,
    reason="ASE not installed",
)


def _water_frame(displacement: float = 0.0) -> Molecule:
    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[
            [0.0 + displacement, 0.0, 0.0],
            [0.957 + displacement, 0.0, 0.0],
            [-0.24, 0.927, 0.0],
        ],
    )


def _three_frame_trajectory() -> tuple[list[Molecule], list[float]]:
    frames = [_water_frame(0.0), _water_frame(0.05), _water_frame(0.10)]
    energies = [-75.0, -75.1, -75.05]
    return frames, energies


class TestSaveTrajectoryXyz:
    """Multi-frame XYZ writer (EXPORT.3): universal-format text file."""

    def test_empty_frames_returns_none(self, tmp_path):
        result = save_trajectory_xyz(tmp_path, frames=[], energies=[])
        assert result is None

    def test_writes_file_at_expected_path(self, tmp_path):
        frames, energies = _three_frame_trajectory()
        out = save_trajectory_xyz(tmp_path, frames=frames, energies=energies)
        assert out is not None
        assert out == tmp_path / "trajectory.xyz"
        assert out.exists()

    def test_correct_atom_count_line_per_frame(self, tmp_path):
        frames, energies = _three_frame_trajectory()
        out = save_trajectory_xyz(tmp_path, frames=frames, energies=energies)
        text = out.read_text(encoding="utf-8")
        # Each of the 3 frames starts with "3\n" (water has 3 atoms).
        assert text.count("\n3\n") + (1 if text.startswith("3\n") else 0) == 3

    def test_energy_in_comment_line(self, tmp_path):
        frames, energies = _three_frame_trajectory()
        out = save_trajectory_xyz(tmp_path, frames=frames, energies=energies)
        text = out.read_text(encoding="utf-8")
        # All three energies must appear, formatted to 10 decimal places
        # ('energy=-75.0000000000 Hartree') so external tools parsing
        # extended-XYZ comment lines pick them up.
        assert "energy=-75.0000000000" in text
        assert "energy=-75.1000000000" in text
        assert "energy=-75.0500000000" in text
        assert text.count("Hartree") == 3

    def test_atom_lines_have_correct_count(self, tmp_path):
        frames, energies = _three_frame_trajectory()
        out = save_trajectory_xyz(tmp_path, frames=frames, energies=energies)
        text = out.read_text(encoding="utf-8")
        # 3 frames × 3 atoms = 9 atom lines.
        atom_lines = [
            line for line in text.splitlines() if line.startswith(("O ", "H "))
        ]
        assert len(atom_lines) == 9

    def test_missing_energy_falls_back_to_frame_label(self, tmp_path):
        frames = [_water_frame(0.0), _water_frame(0.05)]
        # Only one energy supplied; second frame should fall back to "frame 1".
        out = save_trajectory_xyz(tmp_path, frames=frames, energies=[-75.0])
        text = out.read_text(encoding="utf-8")
        assert "energy=-75.0000000000" in text
        assert "frame 1" in text

    def test_xyz_re_readable_by_ase(self, tmp_path):
        if not _ASE_AVAILABLE:
            pytest.skip("ASE required for round-trip read")
        from ase.io import read as _ase_read

        frames, energies = _three_frame_trajectory()
        out = save_trajectory_xyz(tmp_path, frames=frames, energies=energies)
        loaded = _ase_read(str(out), index=":")
        # ASE returns a list of Atoms objects for index=":".
        assert len(loaded) == 3
        assert list(loaded[0].symbols) == ["O", "H", "H"]
        # ASE reads coords back; check they match within float-precision.
        import numpy as _np

        _np.testing.assert_allclose(loaded[0].positions[0], [0.0, 0.0, 0.0], atol=1e-5)


@ase_only
class TestSaveTrajectoryAse:
    """ASE binary Trajectory writer (EXPORT.7)."""

    def test_empty_frames_returns_none(self, tmp_path):
        assert save_trajectory_ase(tmp_path, frames=[], energies=[]) is None

    def test_writes_file_at_expected_path(self, tmp_path):
        frames, energies = _three_frame_trajectory()
        out = save_trajectory_ase(tmp_path, frames=frames, energies=energies)
        assert out is not None
        assert out == tmp_path / "trajectory.traj"
        assert out.exists()

    def test_round_trip_via_ase_trajectory_reader(self, tmp_path):
        from ase.io.trajectory import Trajectory

        frames, energies = _three_frame_trajectory()
        out = save_trajectory_ase(tmp_path, frames=frames, energies=energies)
        # ``Trajectory(path)`` (no mode) opens for reading.
        traj = Trajectory(str(out))
        try:
            assert len(traj) == 3
            atoms0 = traj[0]
            assert list(atoms0.symbols) == ["O", "H", "H"]
        finally:
            traj.close()

    def test_energies_attached_via_calculator(self, tmp_path):
        from ase.io.trajectory import Trajectory

        frames, energies = _three_frame_trajectory()
        out = save_trajectory_ase(tmp_path, frames=frames, energies=energies)
        traj = Trajectory(str(out))
        try:
            # SinglePointCalculator stores energy in eV. The writer
            # converts Hartree → eV at write time, so the round-trip
            # value must match within the 27.211386... factor.
            atoms0 = traj[0]
            energy_ev = atoms0.get_potential_energy()
            # -75 Ha × 27.2114 = -2040.85 eV (approx).
            assert energy_ev == pytest.approx(-75.0 * 27.211386245988, rel=1e-9)
        finally:
            traj.close()

    def test_slicing_works_via_ase_io_read(self, tmp_path):
        # ASE-GUI's "@0:2" syntax maps to ase.io.read(path, index=':2').
        # Confirm the same syntax works on our output.
        from ase.io import read as _ase_read

        frames, energies = _three_frame_trajectory()
        out = save_trajectory_ase(tmp_path, frames=frames, energies=energies)
        first_two = _ase_read(str(out), index=":2")
        assert len(first_two) == 2

    def test_returns_none_when_ase_missing(self, tmp_path, monkeypatch):
        # Simulate ASE being absent by patching the import inside the
        # helper to raise ImportError. The function must return None
        # rather than propagating the exception.
        import builtins

        original_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name.startswith("ase"):
                raise ImportError("simulated: ASE missing")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        frames, energies = _three_frame_trajectory()
        result = save_trajectory_ase(tmp_path, frames=frames, energies=energies)
        assert result is None
