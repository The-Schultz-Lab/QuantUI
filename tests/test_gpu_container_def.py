"""The GPU Apptainer image cannot drift from the package it installs (M-GPU).

The GPU container targets a cluster (NCShare's H200 nodes) where the feedback
loop is an email, a queue, and a scheduled allocation. A mistake that a local
build would surface in seconds costs days there, so the things that can go
quietly wrong are asserted here instead:

- **The version pin falls behind.** ``quantui-gpu.def`` installs a *published*
  release by number. Bump ``pyproject.toml`` without touching the def and every
  subsequent image silently ships the previous release — with no error, because
  the old version installs perfectly well.
- **The CUDA wheel list gets hand-copied.** ``pyproject.toml`` documents why the
  suffixed ``-cuda12x`` wheels are required (the bare PyPI names are source
  sdists needing a local ``nvcc``). A def that lists those three packages
  directly loses that reasoning and stops tracking the extra.
- **The wrong CUDA line.** ``cuda13x`` wheels hard-fail on NCShare's 570-series
  driver, while ``cuda12x`` runs on both it and any 580+ update.
- **Something disables the GPU inside the GPU image.** A baked-in
  ``QUANTUI_DISABLE_GPU`` is the worst failure available here: every job still
  converges to the right energy, on the CPU, silently.

Pure text assertions over the def and scripts — no Apptainer, no GPU, no build.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DEF = REPO / "apptainer" / "quantui-gpu.def"
VERIFY = REPO / "apptainer" / "verify-gpu.sh"
BUILD = REPO / "apptainer" / "build-gpu.sh"
SBATCH = REPO / "apptainer" / "slurm" / "quantui-gpu-test.sbatch"
PYPROJECT = REPO / "pyproject.toml"


@pytest.fixture(scope="module")
def def_text() -> str:
    return DEF.read_text(encoding="utf-8")


def _pinned_version(text: str) -> str:
    """The %arguments default, not a version mentioned in a comment.

    An unanchored search matches the ``--build-arg QUANTUI_VERSION=…`` *example*
    in the comment above the real default, which sits earlier in the file.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        m = re.match(r"QUANTUI_VERSION=([0-9][^\s]*)$", stripped)
        if m:
            return m.group(1)
    raise AssertionError("no QUANTUI_VERSION default found in %arguments")


def _section(text: str, name: str) -> str:
    """Body of a def section. Anchored to a line start — %help prose mentions
    section names, and an unanchored index() lands on that instead."""
    m = re.search(rf"^{re.escape(name)}\s*$", text, re.M)
    assert m is not None, f"no {name} section"
    rest = text[m.end() :]
    nxt = re.search(r"^%[a-z]+\s*$", rest, re.M)
    return rest[: nxt.start()] if nxt else rest


def _project_version() -> str:
    m = re.search(r'^version = "([^"]+)"', PYPROJECT.read_text(encoding="utf-8"), re.M)
    assert m is not None
    return m.group(1)


class TestTheImageTracksThePackage:
    def test_the_pinned_version_matches_pyproject(self, def_text):
        # The failure mode is silent: an older release installs cleanly, so the
        # build goes green and ships the wrong code. Only a comparison catches
        # it. If this fires after a version bump, update the def's %arguments
        # default — and remember the release must be on PyPI before it builds.
        assert _pinned_version(def_text) == _project_version()

    def test_the_version_is_a_real_pin_not_a_range(self, def_text):
        # `pip install quantui==X` — not >=, not bare. An unpinned install makes
        # the image unreproducible, which defeats the point of pinning a release
        # rather than cloning a commit.
        assert '"quantui[' in def_text
        assert ']=={{ QUANTUI_VERSION }}"' in def_text

    def test_the_gpu_extra_is_named_not_expanded(self, def_text):
        # Hand-listing the wheels is how the course container drifted from this
        # package. Naming the extra means pyproject stays the single source.
        assert "gpu-cuda12x" in def_text
        for wheel in ("gpu4pyscf-cuda12x", "cupy-cuda12x", "cutensor-cu12"):
            assert f"pip install --no-cache-dir {wheel}" not in def_text
            # Naming them in a comment is fine; installing them directly is not.
            for line in def_text.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                assert not stripped.startswith(
                    wheel
                ), f"{wheel} installed directly — use the gpu-cuda12x extra"

    def test_the_extras_exist_in_pyproject(self, def_text):
        pyproject = PYPROJECT.read_text(encoding="utf-8")
        m = re.search(r"quantui\[([^\]]+)\]", def_text)
        assert m is not None
        for extra in m.group(1).split(","):
            assert (
                f"{extra.strip()} = [" in pyproject
            ), f"def requests extra '{extra.strip()}' that pyproject does not define"


class TestCudaLineIsCorrectForTheTarget:
    def test_cuda12x_not_cuda13x(self, def_text):
        # NCShare reports CUDA 12.8 on a 570-series driver. CUDA's driver API is
        # backward compatible, so cuda12x runs on 570 AND on any 580+ update;
        # cuda13x would hard-fail on 570. H200 is sm_90 and has prebuilt wheels
        # on both lines, so this is purely about the driver floor.
        assert "gpu-cuda13x" not in def_text
        assert "gpu-cuda12x" in def_text

    def test_base_image_is_a_cuda_12_image(self, def_text):
        m = re.search(r"^From:\s*(\S+)", def_text, re.M)
        assert m is not None
        assert m.group(1).startswith("nvidia/cuda:12."), m.group(1)


class TestNothingSilentlyDisablesTheGpu:
    def test_the_def_never_sets_the_disable_flag(self, def_text):
        # Setting it would make every job converge correctly on the CPU — right
        # answer, wrong device, no error anywhere.
        for line in def_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert not re.match(r"export\s+QUANTUI_DISABLE_GPU=", stripped), stripped

    def test_the_build_asserts_the_flag_is_unset(self, def_text):
        # %test runs on the build host, so it can check this without a GPU.
        test_section = _section(def_text, "%test")
        assert "QUANTUI_DISABLE_GPU" in test_section

    def test_build_time_tests_never_require_a_device(self, def_text):
        # %test runs wherever the build runs — usually a login node or laptop
        # with no GPU. A device assertion there turns "no GPU on the build host"
        # into a failed build, which reads as a broken recipe.
        test_section = _section(def_text, "%test")
        assert "nvidia-smi" not in test_section
        assert "getDeviceCount" not in test_section
        assert "cupy.cuda" not in test_section


class TestTheVerificationLadder:
    @pytest.fixture(scope="class")
    def verify_text(self) -> str:
        return VERIFY.read_text(encoding="utf-8")

    @pytest.mark.parametrize("script", [VERIFY, BUILD, SBATCH])
    def test_scripts_are_syntactically_valid(self, script):
        assert subprocess.run(["bash", "-n", str(script)]).returncode == 0

    def test_the_negative_control_inverts_its_polarity(self, verify_text):
        # `quantui gpu check` exits 1 when offload is unavailable. For step 5
        # that is the PASS condition — the whole point is that the GPU goes
        # away. An early draft treated non-zero as failure there, which would
        # have reported FAIL on a perfectly healthy H200 and burned an
        # allocation chasing it.
        step5 = verify_text[verify_text.index('step "5.') :]
        step5 = step5[: step5.index('step "6.')]
        assert 'if [ "$rc" -ne 0 ]' in step5, "step 5 must pass on a NON-zero exit"

    def test_every_container_call_uses_the_same_gpu_flags(self, verify_text):
        # WSL needs extra binds beyond --nv. If one call is left as a bare
        # --nv it behaves differently from its neighbours, and the ladder's
        # "each step isolates one layer" property quietly breaks.
        assert "exec --nv" not in verify_text
        assert verify_text.count('"${NV_FLAGS[@]}"') >= 5

    def test_the_real_calculation_gates_on_gpu_used(self, verify_text):
        # A fallback still converges to the correct energy, so neither exit
        # status nor the number proves anything. Only gpu_used does.
        assert "r.gpu_used" in verify_text
        assert "SystemExit(0 if r.gpu_used else 1)" in verify_text

    def test_settings_inheritance_is_checked(self, verify_text):
        # Apptainer bind-mounts $HOME, so compute.gpu_enabled=false persisted on
        # any other machine follows the user onto the cluster and silently
        # forces CPU. Found live on the dev box, 2026-08-03.
        assert "gpu_enabled" in verify_text
        assert "QUANTUI_SETTINGS_PATH" in verify_text


class TestTheSlurmTemplate:
    @pytest.fixture(scope="class")
    def sbatch_text(self) -> str:
        return SBATCH.read_text(encoding="utf-8")

    def test_site_specific_values_are_flagged_not_guessed(self, sbatch_text):
        # The partition name and gres string are unconfirmed. A plausible-looking
        # guess is worse than an obvious placeholder: it fails later and less
        # clearly. Remove these TODOs only when the site confirms them.
        assert "CHANGEME" in sbatch_text
        assert sbatch_text.count("TODO(NCShare)") >= 2

    def test_the_exit_status_survives_a_failed_run(self, sbatch_text):
        # `set -e` plus a bare srun would abort before the summary prints —
        # losing exactly the output a failed job exists to produce.
        assert "|| STATUS=$?" in sbatch_text

    def test_threads_are_bound_to_the_allocation(self, sbatch_text):
        # Unset, OpenMP claims every core it can see on a shared node, including
        # other jobs' cores. Applies to GPU runs too — parts of SCF stay on host.
        assert "OMP_NUM_THREADS" in sbatch_text
