#!/usr/bin/env bash
set -euo pipefail

kernel_release="$(uname -r)"

if [[ "$EUID" -ne 0 ]]; then
  echo "Run this script with sudo: sudo bash $0" >&2
  exit 1
fi

echo "Building installed DKMS modules for ${kernel_release}..."
dkms autoinstall -k "$kernel_release"
depmod -a "$kernel_release"

echo "Loading NVIDIA UVM..."
modprobe nvidia_uvm

echo "Installed NVIDIA modules:"
find "/lib/modules/${kernel_release}" -type f -name 'nvidia*.ko*' -print

echo "Loaded NVIDIA modules:"
lsmod | grep '^nvidia'

echo "UVM device nodes:"
ls -l /dev/nvidia-uvm*

echo "NVIDIA UVM repair completed."
