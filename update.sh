#!/usr/bin/env bash
# LiveActionAOV — pull the latest code + re-sync dependencies.
# Idempotent: if nothing changed, it's a no-op.

set -e

cd "$(dirname "$0")"

echo "======================================================================"
echo " LiveActionAOV — update"
echo "======================================================================"
echo

if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] uv is not installed or not on PATH."
    echo "        Run install.sh first."
    exit 1
fi

echo " Fetching latest from origin..."
if ! git pull origin main; then
    echo
    echo "[ERROR] git pull failed. If you have local commits, push or stash them first."
    exit 1
fi
echo

echo " Re-syncing dependencies (no-op if pyproject is unchanged)..."
if ! uv sync --extra all; then
    echo
    echo "[ERROR] uv sync failed. See the output above for details."
    exit 1
fi
echo

echo "----------------------------------------------------------------------"
echo " [OK] LiveActionAOV is up to date."
echo "----------------------------------------------------------------------"
