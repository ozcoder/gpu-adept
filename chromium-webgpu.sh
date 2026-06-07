#!/bin/bash
# Filter out conflicting default args, add WebGPU flags
ARGS=()
for arg in "$@"; do
  case "$arg" in
    --use-angle=*|--use-gl=*|--ozone-platform-hint=*) ;;
    *) ARGS+=("$arg") ;;
  esac
done
exec /usr/lib/chromium/chromium \
  --enable-unsafe-webgpu \
  --ozone-platform=x11 \
  --use-angle=vulkan \
  --enable-features=Vulkan,VulkanFromANGLE \
  "${ARGS[@]}"
