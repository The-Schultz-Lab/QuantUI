"""Write a small set of real QuantUI result folders (water, RHF/B3LYP) for the
browser build's demo content. Needs PySCF; takes ~15 s on a laptop CPU.

    python packaging/viewer/browser/make_samples.py OUT_DIR
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

from quantui.benchmarks import _save_calibration_step
from quantui.freq_calc import run_freq_calc
from quantui.molecule import Molecule
from quantui.optimizer import optimize_geometry
from quantui.results_storage import save_result
from quantui.session_calc import run_in_session
from quantui.tddft_calc import run_tddft_calc


def main(out: Path) -> None:
    import os

    out.mkdir(parents=True, exist_ok=True)
    os.environ["QUANTUI_RESULTS_DIR"] = str(out)
    water = Molecule(
        ["O", "H", "H"], [[0, 0, 0.117], [0, 0.757, -0.469], [0, -0.757, -0.469]]
    )

    def run(fn, calc_type, **kw):
        log = io.StringIO()
        res = fn(water, progress_stream=log, **kw)
        # Same save sequence the app's own calibration runner uses.
        _save_calibration_step(
            res,
            calc_type=calc_type,
            pyscf_log=log.getvalue(),
            calibration_run_id="",
            mol=water,
        )

    run(run_in_session, "single_point", method="B3LYP", basis="6-31G")
    run(optimize_geometry, "geometry_opt", method="RHF", basis="STO-3G")
    run(run_freq_calc, "frequency", method="RHF", basis="STO-3G")
    log = io.StringIO()
    res = run_tddft_calc(
        water, method="B3LYP", basis="6-31G", nstates=5, progress_stream=log
    )
    save_result(
        res,
        pyscf_log=log.getvalue(),
        calc_type="tddft",
        molecule=water,
        spectra={
            "uv_vis": {
                "excitation_energies_ev": res.excitation_energies_ev,
                "oscillator_strengths": res.oscillator_strengths,
                "wavelengths_nm": res.wavelengths_nm(),
            }
        },
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
