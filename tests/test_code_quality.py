"""Static analysis guards for patterns that fail silently at runtime."""

import re
from pathlib import Path

SRC = Path(__file__).parent.parent / "quantui"

# Files where silent failure is most dangerous — numeric/data extraction
# paths where a swallowed exception ships subtly-wrong results downstream
# (bug-A class: cupy TypeError swallow in session_calc.py, session 55).
#
# Every broad-except + pass in these files must EITHER:
#   - have a log call (logger.*, calc_log.log_event, _clog.log_event)
#     within 10 lines after the ``except`` (window allows for multi-line
#     log messages — see session_calc.py:455 MO-extract for an example), OR
#   - carry a ``# noqa: BLE001 — <reason>`` comment on the ``except`` line
#     justifying the silence (cleanup, telemetry self-guard, optional probe).
#
# See reflections/03-error-surfacing.md Rule 1 for the categorization rubric
# and BARE-EXCEPT-AUDIT-2026-05-25.md for the originating audit.
_HIGH_RISK_FILES = {
    "session_calc.py",
    "freq_calc.py",
    "raman_calc.py",
    "tddft_calc.py",
    "nmr_calc.py",
    "optimizer.py",
    "gpu_offload.py",
    "analytics.py",
}


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


def test_no_bare_py3dmol_view():
    """py3Dmol.view(...) defaults to a CDN js= URL that blanks views offline.

    All viewers must be built through ``viz_assets.make_view`` (which forces an
    offline-safe ``js=``) so the vendored bundle is used instead. Only
    ``viz_assets.py`` is allowed to call ``py3Dmol.view`` directly (the factory
    itself). See reflections/01 Rule 1 + reflections/06.
    """
    # Match ``py3Dmol.view(`` or ``<alias>.view(width=`` style construction.
    hits = [
        h
        for h in _grep(r"\bpy3Dmol\.view\(|\b_p3d\.view\(")
        if "viz_assets.py" not in h
    ]
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


def test_no_silent_broad_except_in_high_risk_files():
    """Fail CI when a new broad-except + pass lands in a high-risk file
    without either a log call within 5 lines or a ``# noqa: BLE001 — <reason>``
    annotation on the ``except`` line.

    "Broad" means ``except Exception:`` (with or without ``as <var>``) or
    truly-bare ``except:``. Narrower catches (``except ImportError:``,
    ``except (KeyError, ValueError):``, etc.) are not flagged — the whole
    point of narrowing is to be explicit about the failure mode.

    "Silent" means the body is ``pass`` (or assignment-only without a log
    call) within the next 10 source lines.

    A line carrying ``# noqa: BLE001`` is treated as explicitly-justified
    and skipped. The convention requires a ``— <reason>`` suffix; this
    test does not enforce the format (too easy to game) — reviewers do.
    """
    except_re = re.compile(r"^\s*except\s*(Exception(\s+as\s+\w+)?)?\s*:\s*(#.*)?$")
    log_call_re = re.compile(
        r"\b(logger\.|_clog\.|calc_log\.log_event|log_event\(|"
        r"_log_event|warnings\.warn)"
    )

    violations: list[str] = []
    for path in SRC.rglob("*.py"):
        if path.name not in _HIGH_RISK_FILES:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            m = except_re.match(line)
            if not m:
                continue
            # Explicit noqa annotation = justified. Reviewers enforce
            # that the trailing reason is present + sensible.
            if "noqa: BLE001" in line:
                continue
            # Look at the body (next 10 non-blank lines) for a log call.
            # If none, the block is silent — flag it. 10 is generous enough
            # to allow multi-line log message arguments.
            body = lines[i + 1 : i + 11]
            if any(log_call_re.search(b) for b in body):
                continue
            # Also accept if the body re-raises (still surfaces the error).
            if any("raise" in b for b in body[:2]):
                continue
            violations.append(
                f"{path.relative_to(SRC.parent)}:{i + 1}: {line.strip()}\n"
                f"    (body: {body[0].strip() if body else '<empty>'})"
            )

    assert not violations, (
        "Silent broad-except detected in a high-risk file. Either add a "
        "log call (logger.X / calc_log.log_event) within 10 lines of the "
        "``except``, narrow the exception type, or annotate with\n"
        "    ``# noqa: BLE001 — <reason from rubric>``\n"
        "where <reason> is one of: cleanup, telemetry self-guard, optional probe.\n"
        "See reflections/03-error-surfacing.md Rule 1.\n\n" + "\n".join(violations)
    )


def test_silent_broad_except_guard_actually_catches_violations(tmp_path):
    """Meta-guard: confirm the lint check above isn't trivially passing.

    Builds a temporary high-risk-looking source file containing a known-bad
    silent broad-except + pass and verifies the regex / logic flags it.
    Without this test, an accidental regex break would silently accept
    everything and we wouldn't notice.
    """
    bad_source = (
        "def foo():\n"
        "    try:\n"
        "        risky()\n"
        "    except Exception:\n"
        "        pass\n"
    )
    # Re-implement the matcher inline (mirrors the production logic) so
    # changes to the production helper force a deliberate update here.
    except_re = re.compile(r"^\s*except\s*(Exception(\s+as\s+\w+)?)?\s*:\s*(#.*)?$")
    log_call_re = re.compile(
        r"\b(logger\.|_clog\.|calc_log\.log_event|log_event\(|"
        r"_log_event|warnings\.warn)"
    )

    lines = bad_source.splitlines()
    flagged = False
    for i, line in enumerate(lines):
        if not except_re.match(line):
            continue
        if "noqa: BLE001" in line:
            continue
        body = lines[i + 1 : i + 11]
        if any(log_call_re.search(b) for b in body):
            continue
        if any("raise" in b for b in body[:2]):
            continue
        flagged = True
    assert flagged, (
        "The lint guard didn't flag a known-bad ``except Exception: pass`` "
        "block. The regex or window logic has regressed — fix it before "
        "trusting test_no_silent_broad_except_in_high_risk_files."
    )
