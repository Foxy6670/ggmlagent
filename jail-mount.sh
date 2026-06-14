#!/usr/bin/env bash
# Bind-mount /proc and /dev into the chroot jail after each boot.
# Run with sudo before starting frontier-Boonie:
#   sudo bash jail-mount.sh [mount|umount]
set -euo pipefail

JAIL="/var/ggmlagent-jail"

case "${1:-mount}" in
  mount)
    for fs in proc dev; do
      target="$JAIL/$fs"
      if mountpoint -q "$target"; then
        echo "[jail-mount] $target already mounted, skipping"
      else
        mount --bind "/$fs" "$target"
        echo "[jail-mount] mounted /$fs -> $target"
      fi
    done
    ;;
  umount)
    for fs in dev proc; do
      target="$JAIL/$fs"
      if mountpoint -q "$target"; then
        umount "$target"
        echo "[jail-mount] unmounted $target"
      else
        echo "[jail-mount] $target not mounted, skipping"
      fi
    done
    ;;
  *)
    echo "Usage: $0 [mount|umount]" >&2
    exit 1
    ;;
esac
