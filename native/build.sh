#!/usr/bin/env bash
# 编译 C++ ICP 为 Python 扩展（pybind11 + Eigen + nanoflann，全 header-only）。
# 产物 kitti_slam/icp_cpp*.so，import kitti_slam.icp_cpp 即用。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
PY="${PY:-python3}"

INCLUDES="$($PY -m pybind11 --includes)"
EXT="$($PY -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
OUT="$ROOT/kitti_slam/icp_cpp$EXT"

# nanoflann 恒在 third_party；Eigen 用 vendored(third_party/Eigen)或系统(/usr/include/eigen3)
INCS="-I$ROOT/third_party"
[ -d /usr/include/eigen3 ] && INCS="$INCS -I/usr/include/eigen3"

echo "==> compiling native/icp.cpp -> $OUT"
g++ -O3 -Wall -shared -std=c++17 -fPIC -fopenmp $INCLUDES $INCS \
    "$HERE/icp.cpp" -o "$OUT"
echo "==> done: $(basename "$OUT")"
