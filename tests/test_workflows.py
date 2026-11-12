"""Guards on the GitHub Actions workflows (audit INFRA-7, INFRA-9).

CI cannot run these workflows from a test, so this pins the properties that
matter by reading the YAML: least-privilege token permissions, the release
tag/version check (executed here against the real pyproject), no PyPI
publish for pre-releases, and the comment that used to claim SHA pinning.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def _load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _step(job: dict, name_prefix: str) -> dict:
    steps = [s for s in job["steps"] if s.get("name", "").startswith(name_prefix)]
    assert len(steps) == 1, f"expected one step named {name_prefix!r}"
    return steps[0]


@pytest.mark.parametrize("workflow", ["ci.yml", "release.yml"])
def test_default_token_is_read_only(workflow):
    assert _load(workflow).get("permissions") == {"contents": "read"}


def test_publish_jobs_keep_only_the_oidc_permission():
    jobs = _load("release.yml")["jobs"]
    for name in ("publish-testpypi", "publish-pypi"):
        assert jobs[name]["permissions"] == {"id-token": "write"}, name
    assert "permissions" not in jobs["build"]  # inherits contents: read


def test_pypi_publish_skips_prereleases_and_manual_runs():
    cond = _load("release.yml")["jobs"]["publish-pypi"]["if"]
    assert "github.event_name == 'release'" in cond
    assert "!github.event.release.prerelease" in cond


def _run_tag_check(tag: str) -> subprocess.CompletedProcess:
    step = _step(_load("release.yml")["jobs"]["build"], "Check the release tag")
    assert step["if"] == "github.event_name == 'release'"
    assert step["env"] == {"TAG": "${{ github.event.release.tag_name }}"}
    # The workflow probes the freshly built wheel; here this interpreter's
    # quantui stands in for it.
    script = step["run"].replace("/tmp/wheel-check/bin/python", "python")
    shim = Path(sys.executable).parent
    env = dict(os.environ, TAG=tag, PATH=f"{shim}{os.pathsep}{os.environ['PATH']}")
    return subprocess.run(
        ["bash", "-e", "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.skipif(sys.version_info < (3, 11), reason="step uses tomllib (3.11)")
class TestReleaseTagCheck:
    def test_matching_tag_passes(self):
        import quantui

        result = _run_tag_check(f"v{quantui.__version__}")
        assert result.returncode == 0, result.stdout + result.stderr

    @pytest.mark.parametrize("tag", ["v999.0.0", "0.0.0-no-v-prefix"])
    def test_mismatched_tag_fails(self, tag):
        result = _run_tag_check(tag)
        assert result.returncode != 0
        assert "does not match" in result.stdout

    def test_bare_version_without_v_fails(self):
        import quantui

        assert _run_tag_check(quantui.__version__).returncode != 0


def test_dependabot_does_not_claim_sha_pinning():
    text = (REPO_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    assert "pinned to SHA" not in text
