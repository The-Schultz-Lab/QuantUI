"""Static analysis guards for patterns that fail silently at runtime."""

import ast
import re
from pathlib import Path

SRC = Path(__file__).parent.parent / "quantui"

# Silent broad-excepts (audit INFRA-15). Every ``except Exception`` / bare
# ``except`` in the package must EITHER:
#   - have a log call (logger.*, calc_log.log_event, _clog.log_event,
#     warnings.warn) or a ``raise`` inside the handler body, OR
#   - carry a ``# noqa: BLE001 — <reason>`` comment on the ``except`` header
#     justifying the silence (cleanup, telemetry self-guard, optional probe).
#
# The guard used to cover only eight hand-picked numerics files. It now covers
# the whole package: a module not listed below must have no silent
# broad-excepts (so every new module, numerics or not, is guarded from day
# one), and a listed legacy module may not gain more than it had when the
# guard was widened (2026-11-12). Lower a count when you fix some; drop the
# entry at zero. The original high-risk numerics files (session_calc,
# freq_calc, raman_calc, tddft_calc, nmr_calc, optimizer, gpu_offload,
# analytics) and the newer numerics modules (scf_robust, symmetry,
# reorganization_energy, ts_search, ts_guess, population_analysis, the
# engines, ...) are all at zero.
#
# See reflections/03-error-surfacing.md Rule 1 for the categorization rubric
# and BARE-EXCEPT-AUDIT-2026-05-25.md for the originating audit.
_SILENT_BROAD_EXCEPT_BASELINE = {
    "app.py": 51,
    "app_analysis.py": 9,
    "app_builders.py": 1,
    "app_exports.py": 7,
    "app_history.py": 21,
    "app_launcher.py": 1,
    "app_runflow.py": 38,
    "app_slurm.py": 4,
    "app_visualization.py": 51,
    "app_xyz_input.py": 5,
    "benchmarks.py": 2,
    "c_stderr.py": 5,
    "calc_log.py": 2,
    "cli.py": 2,
    "density_fitting.py": 1,
    "inorganic_guards.py": 1,
    "issue_tracker.py": 1,
    "log_utils.py": 5,
    "orbital_visualization.py": 2,
    "pes_scan.py": 1,
    "pubchem.py": 4,
    "results_storage.py": 14,
    "slurm_ingest.py": 1,
    "worker.py": 2,
    "worker_payload.py": 3,
    "xyz_input.py": 1,
}

_LOG_CALL_RE = re.compile(
    r"\b(logger\.|_clog\.|calc_log\.log_event|log_event\(|"
    r"_log_event|warnings\.warn)"
)


def _silent_broad_excepts(source: str) -> list[int]:
    """Line numbers of silent broad ``except`` handlers in *source*.

    AST-based: the handler body is exactly the ``except`` block, so a log
    call that merely follows the block (the old 10-line text window counted
    those) does not excuse it, and a black-wrapped
    ``except (\n    Exception\n):  # noqa: BLE001 — ...`` header is seen.
    """
    lines = source.splitlines()
    hits = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ExceptHandler):
            continue
        broad = node.type is None or (
            isinstance(node.type, ast.Name)
            and node.type.id in ("Exception", "BaseException")
        )
        if not broad:
            continue
        header = lines[node.lineno - 1 : node.body[0].lineno - 1]
        if any("noqa: BLE001" in h for h in header):
            continue
        body = "\n".join(lines[node.body[0].lineno - 1 : node.end_lineno])
        if _LOG_CALL_RE.search(body):
            continue
        if any(isinstance(n, ast.Raise) for b in node.body for n in ast.walk(b)):
            continue
        hits.append(node.lineno)
    return hits


def _grep(pattern: str) -> list[str]:
    hits = []
    for path in SRC.rglob("*.py"):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(pattern, line):
                hits.append(f"{path.relative_to(SRC.parent)}:{i}: {line.strip()}")
    return hits


def test_no_cdn_plotlyjs():
    hits = _grep(r'include_plotlyjs\s*=\s*["\']cdn["\']')
    assert not hits, "CDN plotlyjs detected (fails silently offline):\n" + "\n".join(
        hits
    )


def _enclosing_def(path: Path, lineno: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()[:lineno]
    for line in reversed(lines):
        m = re.match(r"\s*def (\w+)", line)
        if m:
            return m.group(1)
    return ""


def test_in_app_plotly_figures_do_not_inline_plotlyjs():
    """Audit VIZ-2 (amends DEC-010): plotly >= 6 has no "require" mode, so
    ``include_plotlyjs="require"`` inlined the whole ~4.8 MB bundle into every
    figure. In-app figures go through ``plotly_embed.figure_html``
    (``include_plotlyjs=False`` + one lazy per-page bootstrap). Inline
    (``True``/"inline") is only for standalone exported files, and for
    ``plotly_embed``'s own no-app fallback."""
    require = _grep(r'include_plotlyjs\s*=\s*["\']require["\']')
    assert not require, "include_plotlyjs='require' inlines plotly.js:\n" + "\n".join(
        require
    )
    bad = []
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(text, 1):
            if not re.search(r"include_plotlyjs\s*=\s*True\b", line):
                continue
            # plotly_embed: its own no-app fallback. analytics: writes a
            # standalone dashboard file (inlines plotly.js once).
            if path.name in ("plotly_embed.py", "analytics.py"):
                continue
            if "export" in _enclosing_def(path, i):
                continue  # standalone exported file: must be self-contained
            bad.append(f"{path.relative_to(SRC.parent)}:{i}: {line.strip()}")
    assert not bad, "in-app figure inlines plotly.js:\n" + "\n".join(bad)


def test_in_app_figure_payload_is_small():
    """An in-app figure's HTML must not carry plotly.js (< 200 kB)."""
    import ipywidgets as widgets
    import numpy as np

    from quantui import plotly_embed
    from quantui.orbital_visualization import (
        orbital_info_from_arrays,
        plot_orbital_diagram_plotly,
    )

    out = widgets.Output()
    plotly_embed.register_bootstrap_output(out)
    try:
        info = orbital_info_from_arrays(
            np.linspace(-20.0, 2.0, 40), np.r_[np.full(10, 2.0), np.zeros(30)], "X"
        )
        html = plotly_embed.figure_html(
            plot_orbital_diagram_plotly(info), config={"responsive": True}
        )
        assert len(html.encode("utf-8")) < 200_000
        assert "plotly.js v" not in html
        assert "window.__quantuiPlotly" in html  # waits for the bootstrap
        # The page bootstrap was filled exactly once, with the bundle.
        boot = out.outputs[0]["data"]["text/html"]
        assert "plotly.js v" in boot and "if(!window.__quantuiPlotly)" in boot
        plotly_embed.figure_html(plot_orbital_diagram_plotly(info))
        assert len(out.outputs) == 1
    finally:
        plotly_embed._outputs.discard(out)


def _py3dmol_view_calls(source: str) -> list[int]:
    """Line numbers constructing a py3Dmol view directly, under any import
    alias: ``import py3Dmol as p; p.view(...)``, ``from py3Dmol import view``
    (audit INFRA-15: the old regex knew only ``py3Dmol``/``_p3d``)."""
    tree = ast.parse(source)
    modules, funcs = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "py3Dmol":
                    modules.add(a.asname or a.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "py3Dmol":
            for a in node.names:
                if a.name == "view":
                    funcs.add(a.asname or a.name)
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if (
            isinstance(f, ast.Attribute)
            and f.attr == "view"
            and isinstance(f.value, ast.Name)
            and f.value.id in modules
        ) or (isinstance(f, ast.Name) and f.id in funcs):
            hits.append(node.lineno)
    return hits


def test_py3dmol_guard_sees_aliases():
    src = (
        "import py3Dmol as p\n"
        "from py3Dmol import view as v\n"
        "p.view(width=1)\n"
        "v()\n"
        "other.view()\n"
    )
    assert _py3dmol_view_calls(src) == [3, 4]


def test_no_bare_py3dmol_view():
    """py3Dmol.view(...) defaults to a CDN js= URL that blanks views offline.

    All viewers must be built through ``viz_assets.make_view`` (which forces an
    offline-safe ``js=``) so the vendored bundle is used instead. Only
    ``viz_assets.py`` is allowed to call ``py3Dmol.view`` directly (the factory
    itself). See reflections/01 Rule 1 + reflections/06.
    """
    hits = []
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "viz_assets.py":
            continue
        for line in _py3dmol_view_calls(path.read_text(encoding="utf-8")):
            hits.append(f"{path.relative_to(SRC.parent)}:{line}")
    assert not hits, (
        "Direct py3Dmol.view() call found (fetches 3Dmol.js from a CDN, blank "
        "offline). Use quantui.viz_assets.make_view instead:\n" + "\n".join(hits)
    )


def test_no_bare_scf_kernel_outside_rescue_helper():
    """Every SCF-object ``.kernel()`` call must go through the shared
    M-SCF-ROBUST rescue helper (:func:`quantui.scf_robust.run_scf_with_rescue`)
    instead of calling ``mf.kernel(...)`` directly — a bare kernel call has no
    rescue for a marginal/unstable SCF (the exact failure mode that broke a
    real class deployment; see quantui/scf_robust.py's module docstring and
    QuantUI-development-tracking/TODO/roadmaps/52-....md).

    Excludes:
      - ``scf_robust.py`` itself (the shared helper's own kernel() calls —
        the whole point of this module).
      - ``config.py`` (a string TEMPLATE for a standalone script students
        download and run outside QuantUI — the "mf.kernel()" text there,
        including its own embedded rescue helper, is generated-script
        content, not a live call in this process).
    """
    pattern = re.compile(r"\b(?:mf|_mf_d|mf_d)\.kernel\(")
    hits = []
    for path in SRC.rglob("*.py"):
        if path.name in ("scf_robust.py", "config.py"):
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code_part = line.split("#", 1)[0]  # ignore trailing comments
            stripped = line.strip()
            # Skip prose lines (docstrings/comments referencing "mf.kernel()"
            # by name) — a real call is never inside backticks or a comment.
            if "``" in line or stripped.startswith(("#", '"', "'")):
                continue
            if pattern.search(code_part):
                hits.append(f"{path.relative_to(SRC.parent)}:{i}: {line.strip()}")
    assert not hits, (
        "Bare SCF .kernel() call found outside the shared rescue helper — "
        "route through quantui.scf_robust.run_scf_with_rescue instead:\n"
        + "\n".join(hits)
    )


def test_no_bare_except_pass():
    hits = _grep(r"^\s*except\s*(\(\s*\))?\s*:\s*(pass\s*)?$")
    assert not hits, "Bare except/pass detected (swallows all errors):\n" + "\n".join(
        hits
    )


def test_no_new_silent_broad_except():
    """Fail CI when a broad-except lands without a log call, a re-raise or a
    ``# noqa: BLE001 — <reason>`` annotation on the ``except`` header.

    "Broad" means ``except Exception`` / ``except BaseException`` (with or
    without ``as <var>``) or a bare ``except:``. Narrower catches
    (``except ImportError:``, ``except (KeyError, ValueError):``) are not
    flagged — the whole point of narrowing is to be explicit about the
    failure mode. The noqa convention requires a ``— <reason>`` suffix; this
    test does not enforce the format (too easy to game) — reviewers do.
    """
    violations: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        hits = _silent_broad_excepts(path.read_text(encoding="utf-8"))
        allowed = _SILENT_BROAD_EXCEPT_BASELINE.get(path.name, 0)
        if len(hits) > allowed:
            where = ", ".join(str(n) for n in hits)
            violations.append(
                f"{path.relative_to(SRC.parent)}: {len(hits)} silent "
                f"broad-excepts (allowed {allowed}) at lines {where}"
            )

    assert not violations, (
        "Silent broad-except detected. Either add a log call "
        "(logger.X / calc_log.log_event) inside the ``except`` block, narrow "
        "the exception type, or annotate the ``except`` line with\n"
        "    ``# noqa: BLE001 — <reason from rubric>``\n"
        "where <reason> is one of: cleanup, telemetry self-guard, optional probe.\n"
        "See reflections/03-error-surfacing.md Rule 1.\n\n" + "\n".join(violations)
    )


def test_silent_broad_except_baseline_names_real_files():
    names = {p.name for p in SRC.rglob("*.py")}
    stale = sorted(set(_SILENT_BROAD_EXCEPT_BASELINE) - names)
    assert not stale, f"baseline lists files that no longer exist: {stale}"


def test_silent_broad_except_guard_actually_catches_violations():
    """Meta-guard: confirm the lint check above isn't trivially passing."""
    bad = "def foo():\n    try:\n        risky()\n    except Exception:\n        pass\n"
    assert _silent_broad_excepts(bad) == [4]

    # A log call AFTER the handler (same 10 lines, outer indentation) does
    # not excuse it — the old text window accepted this (audit INFRA-15).
    after = (
        "def foo():\n"
        "    try:\n"
        "        risky()\n"
        "    except Exception:\n"
        "        pass\n"
        "    logger.info('done')\n"
    )
    assert _silent_broad_excepts(after) == [4]

    ok = (
        "def foo():\n"
        "    try:\n"
        "        risky()\n"
        "    except Exception as e:\n"
        "        logger.warning('failed: %s', e)\n"
        "    try:\n"
        "        risky()\n"
        "    except Exception:\n"
        "        raise\n"
        "    try:\n"
        "        risky()\n"
        "    except (\n"
        "        Exception\n"
        "    ):  # noqa: BLE001 — cleanup\n"
        "        pass\n"
        "    try:\n"
        "        risky()\n"
        "    except ValueError:\n"
        "        pass\n"
    )
    assert _silent_broad_excepts(ok) == []
    assert _silent_broad_excepts("try:\n    x()\nexcept:\n    y = 1\n") == [3]


# ── Callback wrapping (audit INFRA-15) ───────────────────────────────────────


def _unwrapped_observe_calls(source: str) -> list[int]:
    """Line numbers of ``.observe(handler, ...)`` calls whose handler is not
    passed through ``_safe_cb`` (``app._safe_cb(fn)`` or a local alias bound
    from ``getattr(app, "_safe_cb", ...)`` / ``app._safe_cb``)."""
    tree = ast.parse(source)
    aliases = {"_safe_cb"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
            if not isinstance(target, ast.Name):
                continue
            if isinstance(value, ast.Attribute) and value.attr == "_safe_cb":
                aliases.add(target.id)
            elif (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "getattr"
                and len(value.args) >= 2
                and isinstance(value.args[1], ast.Constant)
                and value.args[1].value == "_safe_cb"
            ):
                aliases.add(target.id)
    hits = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "observe"
        ):
            continue
        handler = node.args[0] if node.args else None
        if handler is None:
            handler = next((k.value for k in node.keywords if k.arg == "handler"), None)
        wrapped = isinstance(handler, ast.Call) and (
            (
                isinstance(handler.func, ast.Attribute)
                and handler.func.attr == "_safe_cb"
            )
            or (isinstance(handler.func, ast.Name) and handler.func.id in aliases)
        )
        if not wrapped:
            hits.append(node.lineno)
    return hits


def test_observe_handlers_go_through_safe_cb():
    """ipywidgets drops exceptions raised in ``.observe()`` handlers with no
    trace in the UI; ``QuantUIApp._safe_cb`` logs them to the calc log. Every
    ``.observe()`` in the package must wrap its handler with it."""
    hits = []
    for path in sorted(SRC.rglob("*.py")):
        for line in _unwrapped_observe_calls(path.read_text(encoding="utf-8")):
            hits.append(f"{path.relative_to(SRC.parent)}:{line}")
    assert not hits, (
        "Raw .observe() handler (exceptions vanish silently). Wrap it: "
        "``w.observe(app._safe_cb(fn), names=...)``:\n" + "\n".join(hits)
    )


def test_observe_guard_catches_raw_handlers():
    src = (
        "w.observe(fn, names='value')\n"
        "w.observe(app._safe_cb(fn), names='value')\n"
        "safe = getattr(app, '_safe_cb', lambda f: f)\n"
        "w.observe(safe(lambda c: g(c)), names='value')\n"
        "w.observe(handler=lambda c: g(c), names='value')\n"
    )
    assert _unwrapped_observe_calls(src) == [1, 5]


# ── Version + vendored-asset consistency (audit INFRA-15) ────────────────────

_ROOT = SRC.parent


def _pyproject_version() -> str:
    text = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project = text.split("[project]", 1)[1].split("\n[", 1)[0]
    m = re.search(r'^version\s*=\s*"([^"]+)"', project, re.M)
    assert m, "no version in [project] of pyproject.toml"
    return m.group(1)


def test_version_consistent_with_pyproject_and_changelog():
    """pyproject version == quantui.__version__ == newest released CHANGELOG
    entry (the first ``## [x.y.z]`` heading after ``## [Unreleased]``)."""
    import quantui

    changelog = (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    released = re.findall(r"^## \[(\d+\.\d+\.\d+[^\]]*)\]", changelog, re.M)
    assert released, "no released ``## [x.y.z]`` entry in CHANGELOG.md"
    pyproject = _pyproject_version()
    assert (
        quantui.__version__ == pyproject
    ), f"quantui.__version__ {quantui.__version__} != pyproject {pyproject}"
    assert (
        released[0] == pyproject
    ), f"top released CHANGELOG entry {released[0]} != pyproject {pyproject}"


def test_vendored_3dmol_matches_py3dmol_default():
    """The vendored 3Dmol.js must match the release line py3Dmol's own
    ``view(js=...)`` default targets: py3Dmol generates JS against that API.
    Major.minor must agree; a patch-level skew is tolerated (audit INFRA-16
    tracks the current 2.5.4 vs 2.5.5 one)."""
    import inspect

    import pytest

    py3Dmol = pytest.importorskip("py3Dmol")
    from quantui import viz_assets

    assert viz_assets.THREEDMOL_VERSION in viz_assets.CDN_URL
    default_js = inspect.signature(py3Dmol.view).parameters["js"].default
    m = re.search(r"3dmol@(\d+)\.(\d+)\.(\d+)", str(default_js), re.I)
    assert m, f"cannot read the 3Dmol.js version from py3Dmol's default {default_js!r}"
    vendored = viz_assets.THREEDMOL_VERSION.split(".")[:2]
    assert vendored == [m.group(1), m.group(2)], (
        f"vendored 3Dmol.js {viz_assets.THREEDMOL_VERSION} vs py3Dmol default "
        f"{'.'.join(m.groups())}: re-vendor data/js/3Dmol-min.js"
    )


def test_precommit_revs_satisfy_dev_extra_pins():
    """``.pre-commit-config.yaml`` (what CI's lint job runs) and the ``dev``
    extra (what a developer's bare ``black``/``ruff``/``mypy`` is) must agree,
    or a local run passes code CI rejects (M-CLEAN CLEAN.2)."""
    from packaging.specifiers import SpecifierSet

    config = (_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    revs = {}
    repo = ""
    for line in config.splitlines():
        m = re.match(r"\s*-\s*repo:\s*(\S+)", line)
        if m:
            repo = m.group(1).rstrip("/").rsplit("/", 1)[-1]
            repo = repo.replace("mirrors-", "").replace("-pre-commit", "")
            continue
        m = re.match(r"\s*rev:\s*v?(\S+)", line)
        if m and repo:
            revs[repo] = m.group(1)
    pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dev = pyproject.split("\ndev = [", 1)[1].split("\n]", 1)[0]
    checked = []
    for tool in ("black", "ruff", "mypy"):
        rev = revs.get(tool)
        assert rev, f"no pre-commit rev found for {tool}"
        # Only the specifier that applies to a supported CI Python (3.10+).
        specs = [
            m.group(1)
            for m in re.finditer(rf'"{tool}([^";]+)(;[^"]*)?"', dev)
            if not m.group(2) or "3.10" in m.group(2)
        ]
        assert specs, f"no {tool} pin in the dev extra"
        assert rev in SpecifierSet(
            specs[0]
        ), f"pre-commit {tool} rev {rev} is outside the dev extra's {specs[0]}"
        checked.append(tool)
    assert checked == ["black", "ruff", "mypy"]
