#!/usr/bin/env bash
# PINE reset — clears project data but keeps downloaded models
# Run from repo root: backend/reset.command  OR  chmod +x backend/reset.command && ./backend/reset.command
#
# Options:
#   --keep-venv   DEV ONLY. Keeps backend/.venv, so the multi-gigabyte ML stack
#                 (torch/torchaudio/whisperx/pyannote, installed by
#                 model_manager.py during onboarding — not by the installer)
#                 survives the reset and onboarding reuses it. Turns a
#                 minutes-long cycle into a seconds-long one.
#                 Because Setup_MAC.command runs its first-time setup only
#                 when .venv is missing, this ALSO skips the install-time file
#                 layout move — so it does not exercise the installer.
#                 Must be OFF for release verification: a release build has to be
#                 reset with no flags, so the installer and the file-layout move
#                 are exercised the way a first-time user hits them.

set -e
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
shopt -s nullglob

KEEP_VENV=0
while [ $# -gt 0 ]; do
    case "$1" in
        --keep-venv)
            KEEP_VENV=1
            ;;
        *)
            echo ""
            echo "  Unknown option: $1"
            echo "  Usage: backend/reset.command [--keep-venv]"
            exit 2
            ;;
    esac
    shift
done

# What survives a reset is listed in tools/reset_preserve_*.txt — the same two
# files reset_win.bat and the in-app reset read, so the three cannot drift
# apart. Everything unlisted is deleted, so a missing or truncated list has to
# stop the run here, before anything is removed.
PRESERVE_ROOT_LIST="$SCRIPT_DIR/tools/reset_preserve_root.txt"
PRESERVE_BACKEND_LIST="$SCRIPT_DIR/tools/reset_preserve_backend.txt"

read_preserve_list() {
    # Drop comments, CR (the lists are checked out CRLF on Windows) and blanks.
    tr -d '\r' < "$1" \
        | sed -e 's/#.*//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
        | grep -v '^$' || true
}

validate_preserve_list() {
    if [ ! -f "$1" ]; then
        echo "[ERROR] Missing allowlist $1 - nothing was deleted." >&2
        exit 1
    fi
    local count
    count="$(read_preserve_list "$1" | grep -c . || true)"
    if [ "$count" -lt 8 ]; then
        echo "[ERROR] $(basename "$1") looks truncated ($count entries) - nothing was deleted." >&2
        exit 1
    fi
}

validate_preserve_list "$PRESERVE_ROOT_LIST"
validate_preserve_list "$PRESERVE_BACKEND_LIST"

ROOT_PRESERVE=()
while IFS= read -r item; do
    ROOT_PRESERVE+=("$item")
done < <(read_preserve_list "$PRESERVE_ROOT_LIST")

BACKEND_PRESERVE=()
while IFS= read -r item; do
    BACKEND_PRESERVE+=("$item")
done < <(read_preserve_list "$PRESERVE_BACKEND_LIST")

if [ "$KEEP_VENV" = "1" ]; then
    BACKEND_PRESERVE+=(".venv")
fi

should_keep() {
    local name="$1"
    shift
    local item
    for item in "$@"; do
        if [ "$name" = "$item" ]; then
            return 0
        fi
    done
    return 1
}

remove_unpreserved_children() {
    local base_dir="$1"
    shift
    local path
    local name
    for path in "$base_dir"/* "$base_dir"/.[!.]* "$base_dir"/..?*; do
        [ -e "$path" ] || continue
        name="$(basename "$path")"
        if should_keep "$name" "$@"; then
            continue
        fi
        echo "  Removing ${path#../}..."
        rm -rf "$path"
    done
}

echo ""
echo "  ============================================================"
echo "  WARNING: This will PERMANENTLY remove ALL project data."
echo "  All projects, recordings, transcripts, and annotations"
echo "  will be deleted. This cannot be undone."
echo "  ============================================================"
echo ""
echo "  Close the app before running."
echo ""
if [ "$KEEP_VENV" = "1" ]; then
    echo "  DEV MODE (--keep-venv): backend/.venv will be PRESERVED."
    echo "  The installer will skip first-time setup, so the install-time"
    echo "  file layout is NOT re-tested by this run."
    echo ""
fi
read -r -p "  Type Yes and press Enter to proceed: " CONFIRM
if [ "$CONFIRM" != "Yes" ]; then
    echo "  Reset cancelled."
    exit 0
fi
echo ""

rm -f ../backend/data/onboarding_complete.flag
rm -f ../"Launch Pine.bat" ../"Launch Pine.command" ../"Launch Pine.vbs" ../"Launch_WIN.bat" ../"Launch_MAC.command"

# Restore dev files moved during install
for f in LICENSE .editorconfig .gitattributes .gitignore; do
  [[ -f "../documentation/dev-config/$f" && ! -f "../$f" ]] && mv "../documentation/dev-config/$f" "../$f" 2>/dev/null || true
done
rmdir ../documentation/dev-config 2>/dev/null || true

if [ ! -f ../Setup_WIN.bat ] && [ -f ./Setup_WIN.bat ]; then
    cp ./Setup_WIN.bat ../Setup_WIN.bat
    chmod +x ../Setup_WIN.bat 2>/dev/null || true
fi
if [ ! -f ../Setup_MAC.command ] && [ -f ./Setup_MAC.command ]; then
    cp ./Setup_MAC.command ../Setup_MAC.command
    chmod +x ../Setup_MAC.command 2>/dev/null || true
fi
rm -f ./Setup_WIN.bat ./Setup_MAC.command

remove_unpreserved_children .. "${ROOT_PRESERVE[@]}"
remove_unpreserved_children . "${BACKEND_PRESERVE[@]}"

# Purge Python bytecode and test caches so reset returns a pristine source tree.
# cwd is backend/ (all Python lives here); models/ and .git/ are left alone.
# With --keep-venv the preserved .venv is pruned: recursing through site-packages
# would churn thousands of caches for no gain and spend exactly the time the flag
# exists to save. rm -f is used instead of -delete because -delete implies -depth,
# which silently disables -prune.
PURGE_PRUNE=()
if [ "$KEEP_VENV" = "1" ]; then
    PURGE_PRUNE=(-path ./.venv -prune -o)
fi
find . "${PURGE_PRUNE[@]}" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
find . "${PURGE_PRUNE[@]}" -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
find . "${PURGE_PRUNE[@]}" -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '*.pyc.*' \) -exec rm -f {} + 2>/dev/null || true

shopt -u nullglob

echo ""
if [ "$KEEP_VENV" = "1" ]; then
    echo "  Reset complete. Models folder and backend/.venv preserved."
    echo "  Onboarding will reuse the ML packages already in the venv."
    echo "  This is a DEV shortcut — run without --keep-venv before a release."
else
    echo "  Reset complete. Models folder preserved."
fi
echo ""
