#!/bin/sh
# Vx links its executable with clang. This wrapper adds the shim object, which
# vxc has no flag for. Set CLANG_PATH to this script and VX_SHIM to the object.
exec clang "$@" "$VX_SHIM"
