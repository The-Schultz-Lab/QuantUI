"""M-TYPECHECK TYPE.5 — guard the CLASS of bug, not just today's instance.

The mypy pre-commit hook was pinned ``stages: [pre-push]`` (kept off every
commit — it's slow), and CI's "Lint & type check" job ran
``pre-commit run --all-files``, which only executes **default-stage** hooks.
So mypy silently never ran: not on commit, not in CI, despite the job name,
the step name, and the branch-protection check all claiming type checking was
happening. Nobody would find this by looking at CI results — a missing job is
visible; a green one that skipped its work is not (see roadmap
42-m-typecheck-restore-type-checking-in-ci).

Fixed by giving mypy its own explicit CI step (.github/workflows/ci.yml) that
invokes it directly, independent of pre-commit's stage filtering. This test
is the regression guard: it walks every hook in ``.pre-commit-config.yaml``
and asserts that any hook restricted away from the default stage is
independently invoked somewhere in the CI workflow — so the *next* hook to
acquire an unusual ``stages:`` pin fails a test instead of vanishing from CI
silently, the way mypy did.

Deliberately dependency-free (no PyYAML import): both config files are simple
enough that line-based parsing is robust and keeps this test from depending
on a package that isn't a direct project dependency.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PRECOMMIT_CONFIG = _REPO_ROOT / ".pre-commit-config.yaml"
_CI_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"

# Modern pre-commit's default stage is named "pre-commit" (the legacy alias
# "commit" is still accepted). A hook with no `stages:` key — or whose
# `stages:` list includes either spelling — runs under a bare
# `pre-commit run --all-files`, with no separate CI step required.
_DEFAULT_STAGE_NAMES = {"pre-commit", "commit"}

_HOOK_ID_RE = re.compile(r"^\s*-\s*id:\s*(\S+)")
_NEXT_BLOCK_RE = re.compile(r"^\s*-\s*(id|repo):")
_STAGES_RE = re.compile(r"^\s*stages:\s*\[(.*?)\]")
_RUN_BLOCK_RE = re.compile(r"^(\s*)run:\s*[|>]?\s*(.*)$")


def _iter_hooks(config_text: str) -> List[Tuple[str, Optional[List[str]]]]:
    """Yield ``(hook_id, stages)`` for each hook block in the pre-commit config.

    ``stages`` is ``None`` when the hook has no ``stages:`` key (i.e. it runs
    at the default stage). Line-based, not a full YAML parse — see the module
    docstring for why.
    """
    lines = config_text.splitlines()
    hooks: List[Tuple[str, Optional[List[str]]]] = []
    i = 0
    while i < len(lines):
        m = _HOOK_ID_RE.match(lines[i])
        if not m:
            i += 1
            continue
        hook_id = m.group(1)
        stages: Optional[List[str]] = None
        j = i + 1
        while j < len(lines) and not _NEXT_BLOCK_RE.match(lines[j]):
            sm = _STAGES_RE.match(lines[j])
            if sm:
                stages = [s.strip().strip("\"'") for s in sm.group(1).split(",")]
                break
            j += 1
        hooks.append((hook_id, stages))
        i = j
    return hooks


def _extract_run_commands(workflow_text: str) -> str:
    """Return just the shell command text from every ``run:`` step body.

    Excludes step ``name:`` labels and comments — a step *named*
    "Type check (mypy)" with an empty or unrelated ``run:`` body must NOT
    count as "mypy is reachable"; only an actually-invoked command should.
    Handles both ``run: <inline command>`` and block-style ``run: |`` /
    ``run: >`` followed by more-indented lines.
    """
    lines = workflow_text.splitlines()
    commands: List[str] = []
    i = 0
    while i < len(lines):
        m = _RUN_BLOCK_RE.match(lines[i])
        if not m:
            i += 1
            continue
        indent, inline = m.group(1), m.group(2)
        if inline:
            commands.append(inline)
        i += 1
        # Block-style body: collect lines indented further than `run:` itself.
        while i < len(lines) and (
            lines[i].strip() == ""
            or len(lines[i]) - len(lines[i].lstrip()) > len(indent)
        ):
            stripped = lines[i].strip()
            if stripped and not stripped.startswith("#"):
                commands.append(stripped)
            i += 1
    return "\n".join(commands)


def test_config_files_exist_and_parse():
    """Sanity check for the two files the reachability test below depends on."""
    assert _PRECOMMIT_CONFIG.exists()
    assert _CI_WORKFLOW.exists()
    hooks = _iter_hooks(_PRECOMMIT_CONFIG.read_text(encoding="utf-8"))
    assert hooks, "no hooks found — the line-based parser may be broken"
    # mypy must currently be present and pinned to a non-default stage — this
    # guards the parser itself against silently matching nothing.
    mypy_stages = dict(hooks).get("mypy")
    assert mypy_stages == ["pre-push"], (
        "expected the mypy hook to still be stages: [pre-push]; if this "
        "changed, review whether the explicit mypy CI step below is still "
        "needed and update this assertion"
    )


def test_every_non_default_stage_hook_is_reachable_from_ci():
    """A hook pinned away from the default stage must be independently
    invoked in CI — never left to rely on the stage-filtered
    ``pre-commit run --all-files`` alone (exactly the bug that hid mypy from
    CI for months)."""
    config_text = _PRECOMMIT_CONFIG.read_text(encoding="utf-8")
    run_commands = _extract_run_commands(_CI_WORKFLOW.read_text(encoding="utf-8"))

    unreachable = []
    for hook_id, stages in _iter_hooks(config_text):
        if stages is None or any(s in _DEFAULT_STAGE_NAMES for s in stages):
            continue  # reached by the default `pre-commit run --all-files`
        # Restricted to a non-default stage: require the tool to appear as an
        # actually-invoked command in CI (not a step `name:` label, not a
        # comment, not merely inside the stage-filtered
        # `pre-commit run --all-files` line, which is what silently skipped
        # mypy for months).
        pattern = re.compile(rf"(?m)^(?!.*pre-commit run).*\b{re.escape(hook_id)}\b")
        if not pattern.search(run_commands):
            unreachable.append((hook_id, stages))

    assert not unreachable, (
        "Hook(s) pinned to a non-default stage but never independently "
        f"invoked in .github/workflows/ci.yml: {unreachable}. "
        "`pre-commit run --all-files` silently skips stage-filtered hooks — "
        "this is exactly what hid mypy from CI (M-TYPECHECK). Add an "
        "explicit CI step that runs the tool directly, or fold it into an "
        "explicit `--hook-stage <name>` invocation that isn't just the bare "
        "`pre-commit run --all-files`."
    )
