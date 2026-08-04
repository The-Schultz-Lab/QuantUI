# _build_env.sh — shared build-environment setup for the Apptainer build scripts.
#
# Sourced by build.sh and build-gpu.sh; not executable on its own. Requires
# $SIF (the output image path) to be set before sourcing.
#
# Everything here exists because of a failure that actually happened, and every
# one of them lands a long way into a slow build:
#
#   - /tmp is a RAM-backed tmpfs on WSL (and small or `nodev` on many systems).
#     Apptainer unpacks the whole rootfs there and writes the squashfs there
#     too, so a build dies with "No space left on device / Probably out of space
#     on output filesystem" while the actual output filesystem has hundreds of
#     GB free. Cost one full CPU-image build, 2026-08-04.
#   - A fixed scratch path lets two builds in one checkout share a rootfs, and
#     the first to exit deletes it under the other. Cost one 15-minute apt fetch.
#   - Apptainer builds as real root under setuid, so an interrupted build leaves
#     root-owned files a normal user cannot delete.
#
# Defines: OWN_TMPDIR, cleanup_tmpdir. Exports APPTAINER_TMPDIR when it picks one.

# ── Refuse to race an in-flight build of the same image ──────────────────────
# flock, not pgrep. A pgrep pattern loose enough to catch `apptainer build`
# also catches the shell that launched it, an editor with the path open, or the
# grep itself — it false-positived on its own invoking command line within
# minutes of being written. A lock file has no such ambiguity.
_LOCKFILE="${SIF}.buildlock"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$_LOCKFILE" || true
  if ! flock -n 9; then
    echo "ERROR: another build of $SIF is already running." >&2
    echo "       (lock: $_LOCKFILE — remove it only if you are sure)" >&2
    exit 1
  fi
else
  echo "Note: flock unavailable — concurrent builds are not guarded." >&2
fi

# ── Pick a scratch directory that can actually hold the build ────────────────
# Redirect away from /tmp when it is RAM-backed (tmpfs), mounted nodev, or
# simply too small. mktemp -d, never a fixed path, so the cleanup below can only
# ever remove its own.
OWN_TMPDIR=""
if [[ -z "${APPTAINER_TMPDIR:-}" ]]; then
  _tmp_fstype="$(findmnt -no FSTYPE /tmp 2>/dev/null || echo "")"
  _tmp_opts="$(findmnt -no OPTIONS /tmp 2>/dev/null || echo "")"
  _tmp_avail_g="$(df -BG --output=avail /tmp 2>/dev/null | tail -1 | tr -dc '0-9')"
  _why=""
  [[ "$_tmp_fstype" == "tmpfs" ]] && _why="/tmp is tmpfs (RAM-backed)"
  [[ "$_tmp_opts" == *nodev* ]] && _why="${_why:+$_why, }/tmp is nodev"
  [[ -n "$_tmp_avail_g" && "$_tmp_avail_g" -lt 25 ]] &&
    _why="${_why:+$_why, }only ${_tmp_avail_g}G free on /tmp"

  if [[ -n "$_why" ]]; then
    # NEVER inside the source tree. quantui.def does `%files . /opt/quantui`,
    # so a scratch dir under $PWD becomes part of what Apptainer copies, and
    # the build dies with "cannot copy a directory, '.', into itself" — plus
    # permission errors from reading the half-built rootfs it is copying.
    # (Put it in $PWD first; it broke exactly this way, 2026-08-04.)
    _scratch_root="${XDG_CACHE_HOME:-$HOME/.cache}/quantui-apptainer-build"
    mkdir -p "$_scratch_root"
    APPTAINER_TMPDIR="$(mktemp -d "$_scratch_root/run-XXXXXX")"
    export APPTAINER_TMPDIR
    OWN_TMPDIR="$APPTAINER_TMPDIR"
    echo "Note: $_why — building in $APPTAINER_TMPDIR instead."
    echo "      Override by setting APPTAINER_TMPDIR yourself."
  fi
fi

# ── Clean up scratch on ANY exit ─────────────────────────────────────────────
cleanup_tmpdir() {
  # Disarm first: with EXIT+INT+TERM armed, a signal otherwise runs this once
  # for the signal and again for the exit it causes, printing the notice twice.
  trap - EXIT INT TERM
  [[ -n "${_LOCKFILE:-}" ]] && rm -f "$_LOCKFILE" 2>/dev/null
  [[ -n "$OWN_TMPDIR" && -d "$OWN_TMPDIR" ]] || return 0
  rm -rf "$OWN_TMPDIR" 2>/dev/null && return 0
  echo >&2
  echo "NOTE: build scratch left behind and is root-owned (Apptainer builds as" >&2
  echo "      root under setuid). Remove it with:" >&2
  echo "        sudo rm -rf '$OWN_TMPDIR'" >&2
}
# INT and TERM as well as EXIT: bash does not reliably run an EXIT trap when it
# is killed while waiting on a child, which is exactly the Ctrl-C case.
trap cleanup_tmpdir EXIT INT TERM

# ── Warn early when any relevant filesystem is short ─────────────────────────
# $APPTAINER_CACHEDIR defaults to ~/.apptainer, which on a cluster is a quota'd
# NFS home far smaller than a container build needs. Redirect it there:
#   export APPTAINER_CACHEDIR=/work/$USER/.apptainer
_free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }
for _dir_desc in "${APPTAINER_TMPDIR:-/tmp}|build scratch (APPTAINER_TMPDIR)" \
                 "${APPTAINER_CACHEDIR:-$HOME/.apptainer}|layer cache (APPTAINER_CACHEDIR)" \
                 "$PWD|output directory"; do
  _dir="${_dir_desc%%|*}"; _desc="${_dir_desc##*|}"
  while [[ ! -d "$_dir" && "$_dir" != "/" ]]; do _dir="$(dirname "$_dir")"; done
  _avail="$(_free_gb "$_dir")"
  if [[ -n "$_avail" && "$_avail" -lt 20 ]]; then
    echo "WARNING: only ${_avail}G free on $_dir — $_desc" >&2
    echo "         A container build wants ~20G here. Redirect it if this is a" >&2
    echo "         quota'd home or a small tmpfs." >&2
  fi
done
