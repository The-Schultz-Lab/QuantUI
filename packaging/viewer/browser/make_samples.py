"""Write a small set of real QuantUI result folders (water, RHF/B3LYP) for the
browser build's demo content. Needs PySCF; takes ~15 s on a laptop CPU.

    python packaging/viewer/browser/make_samples.py OUT_DIR

Each folder is saved the way the app saves a run (same payloads, orbitals,
trajectory, thumbnail), so every Analysis panel a viewer can show has data.
The Frequency run uses the optimized geometry: frequencies (and the
thermochemistry built on them) only mean something at a stationary point.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

from quantui.backends.worker_payload import tddft_result_payload
from quantui.benchmarks import _save_calibration_step
from quantui.freq_calc import run_freq_calc
from quantui.molecule import Molecule
from quantui.optimizer import optimize_geometry
from quantui.results_storage import (
    load_result,
    save_orbitals,
    save_result,
    save_thumbnail,
)
from quantui.session_calc import run_in_session
from quantui.tddft_calc import run_tddft_calc


def main(out: Path) -> None:
    import os

    out.mkdir(parents=True, exist_ok=True)
    os.environ["QUANTUI_RESULTS_DIR"] = str(out)
    water = Molecule(
        ["O", "H", "H"], [[0, 0, 0.117], [0, 0.757, -0.469], [0, -0.757, -0.469]]
    )

    def run(fn, calc_type, mol, **kw):
        log = io.StringIO()
        res = fn(mol, progress_stream=log, **kw)
        # Same save sequence the app's own calibration runner uses.
        _save_calibration_step(
            res,
            calc_type=calc_type,
            pyscf_log=log.getvalue(),
            calibration_run_id="",
            mol=mol,
        )
        return res

    run(run_in_session, "single_point", water, method="B3LYP", basis="6-31G")
    opt = run(optimize_geometry, "geometry_opt", water, method="RHF", basis="STO-3G")
    run(run_freq_calc, "frequency", opt.molecule, method="RHF", basis="STO-3G")

    log = io.StringIO()
    res = run_tddft_calc(
        water, method="B3LYP", basis="6-31G", nstates=5, progress_stream=log
    )
    saved = save_result(
        res,
        pyscf_log=log.getvalue(),
        calc_type="tddft",
        molecule=water,
        spectra=tddft_result_payload(res)["spectra"],
    )
    save_orbitals(saved, res)
    save_thumbnail(saved, load_result(saved))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
