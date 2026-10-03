"""
Install the host-side ``quantui-batch`` launcher (``quantui install-launcher``).

On a cluster, QuantUI lives inside an Apptainer image and ``sbatch`` lives on
the host. Students submit over SSH from the login node, where starting the
image and importing QuantUI's scientific stack is slow, counts as computing
on a shared machine, and has failed (numpy against the per-user thread
limit). So the launcher never touches the image: it is a standard-library
Python script that writes the same job folder ``SlurmBackend.prepare()``
writes, then calls ``sbatch``. The calculation runs in the image on a
compute node.

Everything the launcher must agree with QuantUI on (element table, method
list, limits, estimate factors, the batch-script template, the image's own
Python) is copied in here, at install time, from the QuantUI doing the
installing; the small amount of logic is a port that
``tests/test_batch_submit.py`` keeps equal to QuantUI's. Install it once per
image, from inside that image, so the two always match::

    # instructor, once per image, from an allocation (not the login node):
    apptainer exec /opt/apps/containers/users/quantui.sif \\
        quantui install-launcher /opt/apps/containers/users/bin
"""

from __future__ import annotations

import json
import os
import sys
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Optional

LAUNCHER_NAME = "quantui-batch"


def running_image() -> Optional[str]:
    """Path of the Apptainer/Singularity image this process runs in, if any."""
    for var in ("APPTAINER_CONTAINER", "SINGULARITY_CONTAINER"):
        value = os.environ.get(var)
        if value:
            return value
    return None


def site_constants() -> Dict[str, Any]:
    """The QuantUI values the launcher's port of the prepare step relies on."""
    from quantui import config
    from quantui.backends import cluster_config as cfg
    from quantui.backends import slurm_utils
    from quantui.backends.base import CALC_TYPES
    from quantui.backends.batch_input import PREOPT_CALC_TYPES, SOLVENT_CALC_TYPES
    from quantui.freq_ir_workers import freq_parallel_opt_in

    return {
        "atomic_numbers": dict(config.ATOMIC_NUMBERS),
        "supported_methods": list(config.SUPPORTED_METHODS),
        "default_method": config.DEFAULT_METHOD,
        "default_basis": config.DEFAULT_BASIS,
        "calc_types": list(CALC_TYPES),
        "preopt_calc_types": sorted(PREOPT_CALC_TYPES),
        "calc_tags": dict(slurm_utils._CALC_TYPE_TAGS),
        "basis_factors": dict(slurm_utils.BASIS_FACTORS),
        "calc_factors": dict(slurm_utils.CALC_FACTORS),
        "job_name_max_len": slurm_utils._JOB_NAME_MAX_LEN,
        "slurm_job_name_max_len": slurm_utils.SLURM_JOB_NAME_MAX_LEN,
        "min_cores": cfg.MIN_CORES,
        "max_cores": cfg.MAX_CORES,
        "min_memory_gb": cfg.MIN_MEMORY_GB,
        "max_memory_gb": cfg.MAX_MEMORY_GB,
        "walltime_options": list(cfg.WALLTIME_OPTIONS),
        "default_partition": cfg.DEFAULT_PARTITION,
        "default_mail_events": list(cfg.DEFAULT_MAIL_EVENTS),
        "script_template": cfg.SLURM_SCRIPT_TEMPLATE,
        "attempt_setup_body": cfg._ATTEMPT_SETUP_BODY,
        # The image's environment decides this for jobs run in it (the CPU
        # image sets QUANTUI_FREQ_PARALLEL=1), and the estimate must match.
        "freq_parallel": bool(freq_parallel_opt_in()),
        "solvent_options": sorted(config.SOLVENT_OPTIONS),
        "solvent_calc_types": sorted(SOLVENT_CALC_TYPES),
    }


def render_launcher(image: str, image_python: str, version: str) -> str:
    """The launcher script text with its install-time values filled in."""
    template = (
        resources.files("quantui")
        .joinpath("data")
        .joinpath("launcher")
        .joinpath("quantui_batch.py")
        .read_text(encoding="utf-8")
    )
    site = json.dumps(site_constants(), sort_keys=True)
    return (
        template.replace("@QUANTUI_IMAGE@", image)
        .replace("@QUANTUI_IMAGE_PYTHON@", image_python)
        .replace("@QUANTUI_VERSION@", version)
        .replace('"@QUANTUI_SITE@"', repr(site))
    )


def install_launcher(
    dest_dir: Path, *, image: Optional[str] = None, force: bool = False
) -> int:
    """Write ``dest_dir/quantui-batch``; return a CLI exit code."""
    from quantui import __version__

    image = image or running_image()
    if not image:
        print(
            "quantui install-launcher: run this inside the QuantUI image so the "
            "launcher knows which image to use, e.g.\n"
            "  apptainer exec /path/to/quantui.sif quantui install-launcher\n"
            "or pass --image /path/to/quantui.sif",
            file=sys.stderr,
        )
        return 1
    image = str(Path(image).expanduser())

    target = dest_dir / LAUNCHER_NAME
    if target.exists() and not force:
        print(
            f"quantui install-launcher: {target} already exists "
            "(pass --force to replace it)",
            file=sys.stderr,
        )
        return 1

    dest_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(
        render_launcher(image, sys.executable, __version__), encoding="utf-8"
    )
    target.chmod(0o755)

    print(f"Wrote {target}")
    print(f"  image:            {image}")
    print(f"  python in image:  {sys.executable}")
    print("  job folders:      each user's own ~/.quantui/staging (or their setting)")
    on_path = str(dest_dir.resolve()) in {
        str(Path(p).expanduser().resolve())
        for p in os.environ.get("PATH", "").split(os.pathsep)
        if p
    }
    if not on_path:
        print(
            f"\nUsers add {dest_dir} to their PATH once with:\n"
            f"  echo 'export PATH=\"{dest_dir}:$PATH\"' >> ~/.bashrc && "
            "source ~/.bashrc"
        )
    print(f"\nThen, on the login node:  {LAUNCHER_NAME} help")
    return 0
