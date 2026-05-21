#!/usr/bin/env bash
# WSL pyc co_filename audit
#
# Runs the full pytest suite under WSL (conda env: quantui), then marshal-scans
# tests/**/*.pyc to count Windows-style vs WSL-style `co_filename` strings and
# detect any stale `repos-DEVS/QuantUI-local` references.
#
# Purpose / context: see GOTCHAS.md → "Windows + WSL / pytest co_filename path style".
# Recurring use case: after switching between Windows-host and WSL-host pytest runs,
# confirm the regenerated pyc cache reflects the current execution environment.
#
# Usage:
#   bash scripts/wsl_pyc_audit.sh              # audit existing pyc cache
#   bash scripts/wsl_pyc_audit.sh --wipe       # wipe tests/**/__pycache__ first, then audit

WIPE=0
for arg in "$@"; do
  case "$arg" in
    --wipe|-w)
      WIPE=1
      ;;
    -h|--help)
      sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $arg" >&2
      echo "Usage: $0 [--wipe]" >&2
      exit 2
      ;;
  esac
done

set -u -o pipefail
cd /mnt/c/Users/schul/Documents/local-code-dir/repos-PUBLIC/QuantUI

if [ "$WIPE" = "1" ]; then
  find tests -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null
  find tests -type f -name '*.pyc' -delete 2>/dev/null
  echo "--- tests pycache wiped ---"
fi

start_epoch=$(date +%s.%N)
export START_EPOCH="$start_epoch"
interp="python"
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
  if conda env list 2>/dev/null | awk 'NR>2 {gsub(/\*/,"",$1); if($1!="") print $1}' | grep -qx "quantui"; then
    conda activate quantui >/dev/null 2>&1 || true
  fi
fi
if ! command -v "$interp" >/dev/null 2>&1; then
  interp="python3"
fi
pytest_out=$(mktemp)
$interp -m pytest tests/ -q --no-cov -o addopts='' >"$pytest_out" 2>&1
pytest_rc=$?
pytest_summary=$(grep -E '([0-9]+ (passed|failed|skipped|xfailed|xpassed|error|errors))|=+ .* in [0-9.]+s' "$pytest_out" | tail -n 1)
if [ -z "$pytest_summary" ]; then
  pytest_summary=$(tail -n 1 "$pytest_out")
fi
printf "pytest_summary=%s\n" "$pytest_summary"
printf "pytest_exit_code=%s\n" "$pytest_rc"
rm -f "$pytest_out"
$interp - <<'PY'
from pathlib import Path
import marshal
import os

root = Path('.').resolve()
tests = root / 'tests'
start_epoch = float(os.environ.get('START_EPOCH', '0'))
stale_needles = ('repos-DEVS/QuantUI-local', 'repos-DEVS\\\\QuantUI-local')

def read_co_filename(path: Path) -> str:
    try:
        with path.open('rb') as f:
            f.read(16)
            code_obj = marshal.load(f)
        return getattr(code_obj, 'co_filename', '') or ''
    except Exception:
        return ''

pycs = [p for p in tests.rglob('*.pyc') if p.parent.name == '__pycache__']
stale = 0
win_style = 0
wsl_style = 0
for p in pycs:
    cof = read_co_filename(p)
    if any(n in cof for n in stale_needles):
        stale += 1
    if cof.startswith('C:\\\\'):
        win_style += 1
    if cof.startswith('/mnt/c/'):
        wsl_style += 1

vis_pycs = [p for p in pycs if p.name.startswith('test_visualization_integration')]
if vis_pycs:
    latest = max(vis_pycs, key=lambda p: p.stat().st_mtime)
    sample_cof = read_co_filename(latest)
else:
    sample_cof = ''

outside_touched = []
for d in root.rglob('__pycache__'):
    rel = d.relative_to(root).as_posix()
    if rel.startswith('tests/'):
        continue
    try:
        if d.stat().st_mtime >= start_epoch:
            outside_touched.append(rel)
    except FileNotFoundError:
        pass

print(f'total_pyc_scanned={len(pycs)}')
print(f'stale_count_old_clone={stale}')
print(f'count_windows_style_paths={win_style}')
print(f'count_wsl_paths={wsl_style}')
print(f'sample_co_filename={sample_cof}')
print(f'touched_pycache_outside_tests_count={len(set(outside_touched))}')
PY
