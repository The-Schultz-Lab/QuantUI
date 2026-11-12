"""CPU Apptainer image deployment defaults (quantui.def).

Lightweight text assertions — no Apptainer build required.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CPU_DEF = REPO / "apptainer" / "quantui.def"
GPU_DEF = REPO / "apptainer" / "quantui-gpu.def"


@pytest.fixture(scope="module")
def cpu_def_text() -> str:
    return CPU_DEF.read_text(encoding="utf-8")


def _section(text: str, name: str) -> str:
    m = re.search(rf"^{re.escape(name)}\s*$", text, re.M)
    assert m is not None, f"no {name} section"
    rest = text[m.end() :]
    nxt = re.search(r"^%[a-z]+\s*$", rest, re.M)
    return rest[: nxt.start()] if nxt else rest


class TestCpuTeachingImageDefaults:
    def test_cpu_image_enables_ir_parallel_by_default(self, cpu_def_text):
        env = _section(cpu_def_text, "%environment")
        assert re.search(
            r'export\s+QUANTUI_FREQ_PARALLEL="\$\{QUANTUI_FREQ_PARALLEL:-1\}"', env
        )

    @pytest.mark.parametrize("def_path", [CPU_DEF, GPU_DEF], ids=["cpu", "gpu"])
    def test_results_dir_is_a_default_the_caller_can_override(self, def_path):
        # M-BATCH2 B2.12: a hard export clobbered a site's or job's own value.
        env = _section(def_path.read_text(encoding="utf-8"), "%environment")
        assert re.search(
            r'export\s+QUANTUI_RESULTS_DIR="\$\{QUANTUI_RESULTS_DIR:-\$\{HOME\}/\.quantui/results\}"',
            env,
        )

    def test_cpu_environment_exports_only_overridable_quantui_defaults(
        self, cpu_def_text
    ):
        env = _section(cpu_def_text, "%environment")
        for name, value in re.findall(r"export\s+(QUANTUI_\w+)=(\S+)", env):
            assert value.startswith(f'"${{{name}:-'), (name, value)

    def test_gpu_image_does_not_inherit_cpu_ir_parallel_default(self):
        gpu_text = GPU_DEF.read_text(encoding="utf-8")
        env = _section(gpu_text, "%environment")
        assert "QUANTUI_FREQ_PARALLEL" not in env


class TestCpuImagePySCFExtras:
    """ISSUE.18 — the CPU image must carry what the [pyscf] extra carries."""

    def test_quantui_is_installed_with_the_pyscf_extra(self, cpu_def_text):
        post = _section(cpu_def_text, "%post")
        # A bare `pip install -e .` skipped pyscf-properties (NMR, analytical
        # Raman) and pyscf-dispersion (PBE-D3).
        assert re.search(
            r'pip install[^\n]*-e\s+"?\.\[(?:[\w-]+,)*pyscf\b[^\]]*\]"?', post
        )
        assert not re.search(r"pip install[^\n]*-e\s+\.\s*$", post, re.M)

    def test_both_images_ship_sella_for_transition_states(self, cpu_def_text):
        # M-TS TS.5: Transition State runs in batch jobs and in the window on
        # the cluster, so both images install the [ts] extra and fail the
        # build when Sella does not import.
        for text in (cpu_def_text, GPU_DEF.read_text(encoding="utf-8")):
            post = _section(text, "%post")
            assert re.search(
                r'pip install[^\n]*-e\s+"?\.\[[^\]]*\bts\b[^\]]*\]"?', post
            )
            assert "sella_available()" in post

    def test_build_fails_without_nmr_and_d3(self, cpu_def_text):
        post = _section(cpu_def_text, "%post")
        assert "import pyscf.prop.nmr" in post
        assert "import pyscf.dispersion" in post

    def test_the_extra_still_lists_both_packages(self):
        text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
        extra = text.split("\npyscf = [", 1)[1].split("]", 1)[0]
        assert '"pyscf-properties"' in extra and "pyscf-dispersion" in extra


class TestCpuImageHygiene:
    """Audit HPC-16 — text checks only; nothing here builds or runs the image."""

    def test_pythonpath_has_no_empty_entry(self, cpu_def_text):
        # "/opt/quantui:${PYTHONPATH}" with PYTHONPATH unset ends in ":", an
        # empty entry = the current directory on sys.path (a student's
        # json.py would shadow the standard library).
        env = _section(cpu_def_text, "%environment")
        lines = [ln.strip() for ln in env.splitlines() if "PYTHONPATH=" in ln]
        assert lines == [
            'export PYTHONPATH="/opt/quantui${PYTHONPATH:+:${PYTHONPATH}}"'
        ]

    def test_voila_comes_from_the_app_extra_with_its_bounds(self, cpu_def_text):
        post = _section(cpu_def_text, "%post")
        assert re.search(r'pip install[^\n]*-e\s+"\.\[[^\]]*\bapp\b[^\]]*\]"', post)
        # No unpinned voila install alongside it (continuation lines included).
        unpinned = re.findall(
            r"pip install --no-cache-dir \\\n((?:\s+\S+ \\\n)*\s+\S+)", post
        )
        assert unpinned and all("voila" not in block for block in unpinned)

    def test_build_records_a_freeze(self, cpu_def_text):
        post = _section(cpu_def_text, "%post")
        assert "pip freeze > /opt/build-info/pip-freeze.txt" in post

    def test_stale_reference_sbatch_is_gone(self):
        # It had drifted from the generated submit.slurm (no attempt dir,
        # --nv forced); the generated script is the reference now.
        assert not (REPO / "apptainer" / "slurm" / "quantui-batch.sbatch").exists()
