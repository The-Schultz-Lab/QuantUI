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
        assert re.search(r"export\s+QUANTUI_FREQ_PARALLEL=1\b", env)

    def test_gpu_image_does_not_inherit_cpu_ir_parallel_default(self):
        gpu_text = GPU_DEF.read_text(encoding="utf-8")
        env = _section(gpu_text, "%environment")
        assert "QUANTUI_FREQ_PARALLEL" not in env
