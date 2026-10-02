"""
Install the host-side ``quantui-batch`` launcher (``quantui install-launcher``).

On a cluster, QuantUI lives inside an Apptainer image and ``sbatch`` lives on
the host, so neither side can submit a job alone. The launcher is a small
standard-library Python script that runs on the host: it calls
``quantui submit --prepare-only`` inside the image to write the job folder,
then ``sbatch`` outside it. It ships inside the package
(``quantui/data/launcher/quantui_batch.py``) so the launcher a user installs
always matches the image they install it from; nobody has to clone a repo.

Run inside the image, so the launcher records which image to use::

    apptainer exec /path/to/quantui.sif quantui install-launcher   # -> ~/bin
"""

from __future__ import annotations

import os
import sys
from importlib import resources
from pathlib import Path
from typing import Optional

LAUNCHER_NAME = "quantui-batch"


def running_image() -> Optional[str]:
    """Path of the Apptainer/Singularity image this process runs in, if any."""
    for var in ("APPTAINER_CONTAINER", "SINGULARITY_CONTAINER"):
        value = os.environ.get(var)
        if value:
            return value
    return None


def render_launcher(image: str, staging_root: str, version: str) -> str:
    """The launcher script text with its install-time defaults filled in."""
    template = (
        resources.files("quantui")
        .joinpath("data")
        .joinpath("launcher")
        .joinpath("quantui_batch.py")
        .read_text(encoding="utf-8")
    )
    return (
        template.replace("@QUANTUI_IMAGE@", image)
        .replace("@QUANTUI_STAGING_DIR@", staging_root)
        .replace("@QUANTUI_VERSION@", version)
    )


def install_launcher(
    dest_dir: Path, *, image: Optional[str] = None, force: bool = False
) -> int:
    """Write ``dest_dir/quantui-batch``; return a CLI exit code."""
    from quantui import __version__
    from quantui.backends.cluster_config import default_staging_root

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
    staging = str(default_staging_root())
    target.write_text(render_launcher(image, staging, __version__), encoding="utf-8")
    target.chmod(0o755)

    print(f"Wrote {target}")
    print(f"  image:       {image}")
    print(f"  job folders: {staging}")
    on_path = str(dest_dir.resolve()) in {
        str(Path(p).expanduser().resolve())
        for p in os.environ.get("PATH", "").split(os.pathsep)
        if p
    }
    if not on_path:
        print(
            f"\n{dest_dir} is not on your PATH. Add it once with:\n"
            f"  echo 'export PATH=\"{dest_dir}:$PATH\"' >> ~/.bashrc && "
            "source ~/.bashrc"
        )
    print(f"\nThen try:  {LAUNCHER_NAME} help")
    return 0
