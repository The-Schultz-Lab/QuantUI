#!/usr/bin/env bash
# Shared QuantUI startup banner for the shell launchers.
#
# Sourced, not executed:
#     source "$SCRIPT_DIR/_banner.sh"
#     quantui_banner "NATIVE MODE — local conda env, no container" \
#                    "quantui/*.py edits are live; no rebuild needed"
#
# The wordmark is the same figlet art as the in-app run header
# (quantui/log_utils.py:_ASCII_LOGO_LINES). It is duplicated here rather than
# printed by Python on purpose: the banner runs *before* the conda env is
# activated, so there is no interpreter to ask, and a cosmetic banner must never
# be the thing that fails a launch. If the art ever changes, update both.
#
# Windows .bat launchers are deliberately NOT covered — batch escaping of the
# backslashes and pipes in the art is error-prone and cannot be verified from
# this dev environment. They keep their plain-text headers.

quantui_banner() {
    local mode_line="${1:-}"
    local note_line="${2:-}"
    local c_logo="" c_dim="" c_bold="" c_off=""

    # Colour only when stdout is a TTY, so piping or redirecting a launcher's
    # output stays free of escape codes.
    if [ -t 1 ]; then
        c_logo=$'\033[36m'
        c_dim=$'\033[2m'
        c_bold=$'\033[1m'
        c_off=$'\033[0m'
    fi

    printf '%s\n' "${c_logo}"
    cat <<'LOGO'
   ___                    _   _   _ ___
  / _ \ _   _  __ _ _ __ | |_| | | |_ _|
 | | | | | | |/ _` | '_ \| __| | | || |
 | |_| | |_| | (_| | | | | |_| |_| || |
  \__\_\\__,_|\__,_|_| |_|\__|\___/|___|
LOGO
    printf '%s' "${c_off}"
    printf '  %sQuantum Chemistry Interface%s\n' "${c_bold}" "${c_off}"
    [ -n "${mode_line}" ] && printf '  %s%s%s\n' "${c_dim}" "${mode_line}" "${c_off}"
    [ -n "${note_line}" ] && printf '  %s%s%s\n' "${c_dim}" "${note_line}" "${c_off}"
    printf '\n'
    return 0
}
