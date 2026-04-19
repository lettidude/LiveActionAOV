#!/usr/bin/env bash
# LiveActionAOV — installation script
# Creates project-local .venv, installs dependencies, verifies.

set -u

LOG_FILE="install.log"
: > "$LOG_FILE"

log() {
    local msg="$1"
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$timestamp] $msg" | tee -a "$LOG_FILE"
}

fail() {
    local msg="$1"
    local remediation="${2:-}"
    log "✗ FAILED: $msg"
    if [ -n "$remediation" ]; then
        log "                     → $remediation"
    fi
    log "Install incomplete. See $LOG_FILE for full details."
    exit 1
}

ok() { log "✓ $1"; }

log "LiveActionAOV installer started"

# 1. Check / install uv
if ! command -v uv >/dev/null 2>&1; then
    log "uv not found, installing..."
    if ! curl -LsSf https://astral.sh/uv/install.sh | sh >>"$LOG_FILE" 2>&1; then
        fail "Failed to install uv" "Install manually: https://docs.astral.sh/uv/getting-started/installation/"
    fi
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env" 2>/dev/null || source "$HOME/.local/bin/env" 2>/dev/null || true
fi
UV_VERSION="$(uv --version 2>/dev/null || echo 'unknown')"
ok "uv: $UV_VERSION"

# 2. Check NVIDIA driver (Linux only)
if [[ "$(uname)" == "Linux" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1)"
        CUDA_VER="$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '')"
        ok "NVIDIA driver $DRIVER_VER (CUDA $CUDA_VER)"
    else
        log "⚠ WARNING: nvidia-smi not found. GPU passes will not run."
        log "            If you have an NVIDIA GPU, install drivers from https://www.nvidia.com/Download/index.aspx"
    fi
elif [[ "$(uname)" == "Darwin" ]]; then
    log "ℹ macOS detected — will use MPS backend (Apple Silicon) or CPU (Intel). GPU passes may be slow."
fi

# 3. Provision Python 3.11
if ! uv python install 3.11 >>"$LOG_FILE" 2>&1; then
    fail "Failed to provision Python 3.11 via uv" "Check network connectivity and disk space"
fi
ok "Python 3.11 provisioned"

# 4. Sync dependencies. `[tool.uv.sources]` in pyproject pins torch +
#    torchvision to the pytorch-cu124 index on Windows/Linux; macOS
#    falls through to PyPI for MPS support. `uv sync` is therefore
#    authoritative — no separate reinstall step, no risk of a future
#    resync silently downgrading to the CPU wheel.
log "Installing dependencies (this may take several minutes)..."
if ! uv sync --extra dev >>"$LOG_FILE" 2>&1; then
    fail "uv sync failed" "See $LOG_FILE for details. Common causes: network issues, incompatible torch wheel for platform"
fi
ok "Dependencies installed"

# 5. Verify CUDA is actually available on Linux (macOS uses MPS so
#    cuda.is_available() is always False and that's fine).
if [[ "$(uname)" == "Linux" ]] && [[ -n "${DRIVER_VER:-}" ]]; then
    if uv run python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
            >>"$LOG_FILE" 2>&1; then
        ok "torch.cuda.is_available() confirmed"
    else
        log "⚠ WARNING: torch.cuda.is_available() is False despite NVIDIA driver present."
        log "           Check nvidia-smi output; try re-running after a reboot."
    fi
elif [[ "$(uname)" == "Linux" ]]; then
    log "⚠ No NVIDIA driver detected — neural passes will refuse to run."
    log "  Install drivers then re-run this script."
fi

# 6. Smoke test
if ! uv run liveaov --version >>"$LOG_FILE" 2>&1; then
    fail "Smoke test failed — liveaov --version did not succeed" "See $LOG_FILE"
fi
VERSION_OUTPUT="$(uv run liveaov --version 2>&1)"
ok "Smoke test: $VERSION_OUTPUT"

log ""
log "✓ Installation complete."
log ""
log "Next steps:"
log "  uv run liveaov --help          # see available commands"
log "  uv run liveaov-gui             # launch the preparation GUI"
log "  uv run liveaov models pull --all   # pre-download model checkpoints (optional)"
log ""
log "Activate the environment manually with:"
log "  source .venv/bin/activate   (Linux/macOS)"
log "  .venv\\Scripts\\activate     (Windows)"
