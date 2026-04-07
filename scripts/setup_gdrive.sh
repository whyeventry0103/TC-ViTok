#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_gdrive.sh — mount Google Drive on the remote server using rclone
#
# Run this ONCE on the remote server before any training.
# After setup, Google Drive is available at $MOUNT_POINT.
#
# Usage:
#   bash scripts/setup_gdrive.sh            # interactive rclone config
#   bash scripts/setup_gdrive.sh --mount    # mount after config is done
#   bash scripts/setup_gdrive.sh --unmount  # unmount
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

MOUNT_POINT="${MOUNT_POINT:-/mnt/gdrive}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"   # name you gave the remote in rclone config

# ── Install rclone if missing ─────────────────────────────────────────────────
install_rclone() {
    if command -v rclone &>/dev/null; then
        echo "rclone already installed: $(rclone --version | head -1)"
        return
    fi
    echo "Installing rclone..."
    curl -fsSL https://rclone.org/install.sh | sudo bash
    echo "rclone installed: $(rclone --version | head -1)"
}

# ── Configure rclone (interactive) ───────────────────────────────────────────
configure_rclone() {
    echo ""
    echo "========================================================"
    echo " rclone Google Drive configuration"
    echo "========================================================"
    echo " Steps:"
    echo "  1. Run:  rclone config"
    echo "  2. Choose 'n' for a new remote"
    echo "  3. Name it: ${RCLONE_REMOTE}"
    echo "  4. Storage type: 'drive' (Google Drive)"
    echo "  5. Leave client_id and secret blank (use defaults)"
    echo "  6. Scope: '1' (full drive access)"
    echo "  7. Answer 'n' to auto-config (since this is a headless server)"
    echo "     → rclone will print a URL — open it in your LOCAL browser"
    echo "     → Paste the verification code back here"
    echo "  8. Shared drive: 'n'"
    echo "========================================================"
    echo ""
    rclone config
}

# ── Mount Google Drive ────────────────────────────────────────────────────────
mount_drive() {
    if mountpoint -q "${MOUNT_POINT}" 2>/dev/null; then
        echo "Already mounted at ${MOUNT_POINT}"
        return
    fi
    sudo mkdir -p "${MOUNT_POINT}"
    sudo chmod 777 "${MOUNT_POINT}"

    echo "Mounting ${RCLONE_REMOTE}: → ${MOUNT_POINT} ..."
    rclone mount \
        "${RCLONE_REMOTE}:" \
        "${MOUNT_POINT}" \
        --daemon \
        --vfs-cache-mode full \
        --vfs-cache-max-size 50G \
        --buffer-size 256M \
        --dir-cache-time 72h \
        --poll-interval 15s \
        --log-level INFO \
        --log-file /tmp/rclone_mount.log

    sleep 2
    if mountpoint -q "${MOUNT_POINT}"; then
        echo "✓ Mounted at ${MOUNT_POINT}"
        echo ""
        echo "Add this to your shell profile (~/.bashrc or ~/.zshrc):"
        echo "  export DATA_ROOT=${MOUNT_POINT}/tc-vitok-data"
        echo "  export HF_HOME=\${DATA_ROOT}/hf_cache"
    else
        echo "✗ Mount failed — check /tmp/rclone_mount.log"
        exit 1
    fi
}

# ── Unmount ───────────────────────────────────────────────────────────────────
unmount_drive() {
    if mountpoint -q "${MOUNT_POINT}" 2>/dev/null; then
        fusermount -u "${MOUNT_POINT}" && echo "Unmounted ${MOUNT_POINT}"
    else
        echo "${MOUNT_POINT} is not mounted"
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────────
case "${1:-}" in
    --mount)
        install_rclone
        mount_drive
        ;;
    --unmount)
        unmount_drive
        ;;
    "")
        install_rclone
        configure_rclone
        echo ""
        echo "Configuration done. Now run:"
        echo "  bash scripts/setup_gdrive.sh --mount"
        ;;
    *)
        echo "Usage: $0 [--mount | --unmount]"
        exit 1
        ;;
esac
