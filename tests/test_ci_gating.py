"""Audit INFRA-12 — CI gating guards.

Line-based checks on the workflow files and the test harness (no PyYAML, same
reasoning as ``test_ci_hook_reachability.py``):

* the ``network``-marked tests run somewhere (a scheduled workflow), since
  every ``ci.yml`` job deselects them;
* ``pages.yml`` grants ``pages: write`` / ``id-token: write`` to the deploy
  job only, not workflow-wide (where the build job would inherit them);
* the Windows job runs the whole non-PySCF suite — every test file must skip
  cleanly without PySCF rather than being ``--ignore``\\ d;
* xdist workers cap ``OMP_NUM_THREADS`` (conftest).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_WORKFLOWS = _ROOT / ".github" / "workflows"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _top_level_block(text: str, key: str) -> str:
    """Return the indented body under a top-level ``key:`` line ('' if absent)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.rstrip() == f"{key}:":
            body = []
            for nxt in lines[i + 1 :]:
                if nxt and not nxt[0].isspace() and not nxt.startswith("#"):
                    break
                body.append(nxt)
            return "\n".join(body)
    return ""


def test_network_tests_run_on_a_schedule():
    runners = []
    for wf in sorted(_WORKFLOWS.glob("*.yml")):
        text = _read(wf)
        if re.search(r'pytest[^\n]*-m\s+"?network"?(\s|$)', text):
            runners.append((wf.name, text))
    assert runners, "no workflow runs `pytest -m network`"
    assert any(
        "schedule:" in text and "cron:" in text for _, text in runners
    ), "the workflow running `-m network` has no schedule"


def test_pages_write_permissions_are_job_scoped():
    text = _read(_WORKFLOWS / "pages.yml")
    top = _top_level_block(text, "permissions")
    assert "contents: read" in top
    assert "pages: write" not in top
    assert "id-token: write" not in top
    deploy = text.split("\n  deploy:", 1)
    assert len(deploy) == 2, "pages.yml has no deploy job"
    build = deploy[0].split("\n  build:", 1)[1]
    assert "pages: write" not in build and "id-token: write" not in build
    assert "pages: write" in deploy[1] and "id-token: write" in deploy[1]


def test_windows_job_does_not_ignore_test_files():
    text = _read(_WORKFLOWS / "ci.yml")
    job = text.split("\n  test-windows:", 1)[1].split("\n  test-windows-pyfock:")[0]
    runs = [ln for ln in job.splitlines() if ln.strip().startswith("run:")]
    assert any("pytest" in ln for ln in runs)
    assert not any("--ignore" in ln for ln in runs), (
        "the Windows job --ignores a test file; give it a PySCF skip guard "
        "instead so it runs (and skips cleanly) without PySCF"
    )


@pytest.mark.skipif(
    not os.environ.get("PYTEST_XDIST_WORKER"), reason="only meaningful under xdist"
)
def test_xdist_workers_cap_omp_threads():
    assert os.environ.get("OMP_NUM_THREADS")


def test_conftest_caps_omp_threads_for_xdist_workers():
    # -n0 runs can't observe the worker env, so also pin the conftest logic.
    text = _read(_ROOT / "tests" / "conftest.py")
    assert 'os.environ.get("PYTEST_XDIST_WORKER")' in text
    assert 'os.environ.setdefault("OMP_NUM_THREADS", "1")' in text
