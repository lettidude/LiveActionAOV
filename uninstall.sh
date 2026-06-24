#!/usr/bin/env bash
# LiveActionAOV — uninstall script
# Removes project-local .venv. Optionally removes model cache.

set -u

echo "LiveActionAOV uninstaller"
echo ""

if [ -d ".venv" ]; then
    echo "Removing .venv/"
    rm -rf .venv
    echo "✓ .venv removed"
else
    echo "No .venv found in current directory"
fi

CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/live-action-aov"
if [ -d "$CACHE_DIR" ]; then
    echo ""
    echo "Model cache found at: $CACHE_DIR"
    CACHE_SIZE="$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)"
    echo "Size: $CACHE_SIZE"
    read -r -p "Remove model cache? [y/N] " yn
    case "$yn" in
        [Yy]*) rm -rf "$CACHE_DIR"; echo "✓ Model cache removed" ;;
        *) echo "Model cache kept" ;;
    esac
fi

echo ""
echo "Uninstall complete. User sessions and sidecars are untouched."
