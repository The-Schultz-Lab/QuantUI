"""The live calculation log must render in a fixed-width font.

Regression guard for a 2026-07-30 user report: the run header's ASCII wordmark
rendered as garbled letters and its ``Label           : value`` provenance rows
had visibly drifting colons.

Root cause was not the header-building code — the padding is correct and the art
is faithful figlet. ``_APP_CSS``'s system-font rule lists
``.jp-OutputArea-output``, which is exactly the element the streaming log renders
into, so the log inherited a PROPORTIONAL font with ``!important``. Both symptoms
share that single cause, because both depend on fixed-width character cells.

These tests lock in the override and, importantly, the properties that make it
actually win: it must come *after* the sans-serif rule and out-specify it.
"""

from __future__ import annotations

import re

from quantui.app import _APP_CSS
from quantui.log_utils import _ASCII_LOGO_LINES, _row

_LOG_SELECTOR = ".quantui-run-output .jp-OutputArea-output"


class TestLogFontOverride:
    def test_override_exists(self):
        assert _LOG_SELECTOR in _APP_CSS

    def test_override_declares_monospace_important(self):
        block = _APP_CSS[_APP_CSS.index(_LOG_SELECTOR) :]
        block = block[: block.index("}")]
        assert "monospace" in block
        # The rule it must beat is itself !important.
        assert "!important" in block

    def test_override_comes_after_the_sans_serif_rule(self):
        # Equal specificity would make source order decide; here the override is
        # more specific, but if that ever changes, order is the backstop. A rule
        # placed before the sans-serif block would silently lose.
        assert _APP_CSS.index(_LOG_SELECTOR) > _APP_CSS.index("sans-serif !important")

    def test_sans_serif_rule_still_targets_output_area(self):
        # If this ever stops being true the override is dead weight — but more
        # importantly it means someone "fixed" the font globally, and this test
        # should be revisited rather than deleted.
        head = _APP_CSS[: _APP_CSS.index("sans-serif !important")]
        assert ".jp-OutputArea-output" in head


class TestHeaderIsFixedWidthSafe:
    """The header only lines up if these invariants hold in a monospace font."""

    def test_provenance_rows_align_colons(self):
        rows = [
            _row("Calculation ID", "4ba876a5"),
            _row("Timestamp", "2026-07-30 15:42:42 UTC"),
            _row("Host", "JSchultz-Aurora"),
            _row("Device", "GPU"),
            _row("Threads", "OMP_NUM_THREADS=20"),
        ]
        positions = {r.index(":") for r in rows}
        assert len(positions) == 1, f"colons at differing columns: {positions}"

    def test_longest_label_still_fits_the_pad(self):
        # A label at/over the pad width would push its colon out and break the
        # column for every other row.
        long_row = _row("Calculation ID", "x")
        assert long_row.index(":") == _row("Host", "x").index(":")

    def test_logo_has_no_tab_characters(self):
        # A tab renders at an unpredictable width and would shear the art.
        for line in _ASCII_LOGO_LINES:
            assert "\t" not in line

    def test_logo_is_pure_ascii(self):
        # Non-ASCII glyphs are frequently not fixed-width even in a mono font.
        for line in _ASCII_LOGO_LINES:
            assert line.isascii(), f"non-ASCII in logo art: {line!r}"

    def test_logo_lines_do_not_exceed_the_separator_width(self):
        from quantui.log_utils import _SEP

        for line in _ASCII_LOGO_LINES:
            assert len(line) <= len(_SEP)

    def test_logo_spells_quantui(self):
        # Cheap sanity check that the art wasn't replaced by something unrelated:
        # the figlet body should be built only from box-drawing ASCII.
        body = "".join(_ASCII_LOGO_LINES)
        assert re.fullmatch(r"[ _/\\|(),'`]+", body), "unexpected characters in art"
