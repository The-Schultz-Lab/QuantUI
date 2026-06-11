# QM9 bulk-library provenance

- Entries shipped: 1956 (cap 2500; `category=bulk-qm9`, `source=qm9-dft`)
- Source: QM9 / GDB-9 (figshare dataset 978904)
- Tarball sha256: `3a63848ac80691bdb8d41834b575afad345b9300d7a2db0c38adb7f6eaa8360c`
- Geometries: B3LYP/6-31G(2df,p) optimized
- License: CC0 (public domain)
- Citation: Ramakrishnan, Dral, Rupp & von Lilienfeld, *Sci. Data* **1**, 140022 (2014).
- Selection: stratified by Hill formula (≤9 heavy atoms, uncharacterized excluded, InChIKey-deduped vs the curated set), deterministic round-robin.
- Regenerate: `python scripts/build_bulk_library.py --target N`
