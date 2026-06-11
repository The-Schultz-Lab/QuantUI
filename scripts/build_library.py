"""Rebuild the bundled molecule-library SQLite store from the committed manifests.

Usage:
    python scripts/build_library.py

Reads every ``quantui/data/manifests/*.json`` and writes
``quantui/data/library/library.sqlite`` (deterministic, overwrites). Prints the
final entry count + on-disk size and asserts the 10 MB budget (STRUCT.10).

When STRUCT.7 (curated set) and STRUCT.8 (bulk QM9 subset) land, they add more
manifests; this tool is unchanged.
"""

from quantui import molecule_library as ml

_BUDGET_BYTES = 10 * 1024 * 1024  # 10 MB (DEC-015)


def main() -> None:
    path = ml.build_from_manifests()
    size = path.stat().st_size
    n = ml.count()
    print(f"Built {path} — {n} entries, {size / 1024:.1f} KiB")
    print(f"Categories: {', '.join(ml.categories())}")
    if size > _BUDGET_BYTES:
        raise SystemExit(
            f"Library store {size / 1024 / 1024:.2f} MB exceeds the 10 MB budget"
        )


if __name__ == "__main__":
    main()
