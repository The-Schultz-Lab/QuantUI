"""MkDocs snippet includes must resolve inside the checkout (audit INFRA-5).

``mkdocs.yml`` had ``pymdownx.snippets`` ``base_path: [".."]`` and
``docs/changelog.md`` included ``"../CHANGELOG.md"``. pymdownx resolves
``base_path`` against the working directory (``mkdocs build`` runs from the
repo root), so that pointed two levels above the checkout; with
``check_paths`` off the missing snippet was dropped silently and the
published Changelog page came out empty while ``--strict`` stayed green.

This mirrors pymdownx's lookup (absolute base, restricted to stay under it)
without importing mkdocs, which is not a test dependency. Line-based parsing
because ``mkdocs.yml`` carries ``!!python/name`` tags ``yaml.safe_load``
rejects.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_SNIPPET_RE = re.compile(r"""^-{1,}8<-{1,}\s+["']([^"']+)["']""", re.MULTILINE)


def _snippets_config() -> str:
    text = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    block = re.search(r"- pymdownx\.snippets:\n((?:\s{6,}.*\n)+)", text)
    assert block, "pymdownx.snippets is not configured in mkdocs.yml"
    return block.group(1)


def _base_paths() -> list[Path]:
    m = re.search(r"base_path:\s*\[(.*?)\]", _snippets_config())
    assert m, "pymdownx.snippets has no base_path list"
    bases = [b.strip().strip("\"'") for b in m.group(1).split(",") if b.strip()]
    return [Path(os.path.abspath(REPO_ROOT / b)) for b in bases]


def _resolve(include: str) -> Path | None:
    for base in _base_paths():
        candidate = Path(os.path.abspath(base / include))
        # restrict_base_path (pymdownx default): must stay under the base.
        if not str(candidate).startswith(str(base)):
            continue
        if candidate.is_file():
            return candidate
    return None


def test_check_paths_is_on():
    assert re.search(r"check_paths:\s*true", _snippets_config()), (
        "without check_paths a missing snippet renders as nothing and "
        "`mkdocs build --strict` cannot catch it"
    )


def test_every_docs_snippet_resolves_inside_the_checkout():
    includes = [
        (md.relative_to(REPO_ROOT), inc)
        for md in sorted((REPO_ROOT / "docs").rglob("*.md"))
        for inc in _SNIPPET_RE.findall(md.read_text(encoding="utf-8"))
    ]
    assert includes, "expected at least the changelog snippet include"
    for md, inc in includes:
        resolved = _resolve(inc)
        assert resolved is not None, f"{md}: snippet {inc!r} does not resolve"
        assert resolved.is_relative_to(REPO_ROOT), f"{md}: {inc!r} -> {resolved}"


def test_changelog_page_includes_the_root_changelog():
    text = (REPO_ROOT / "docs" / "changelog.md").read_text(encoding="utf-8")
    (inc,) = _SNIPPET_RE.findall(text)
    assert _resolve(inc) == REPO_ROOT / "CHANGELOG.md"
