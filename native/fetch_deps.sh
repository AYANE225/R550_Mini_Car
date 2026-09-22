#!/usr/bin/env bash
# 拉取 C++ ICP 的 header-only 依赖到 third_party/（nanoflann + Eigen）。
# 已 vendored nanoflann.hpp；Eigen 体积大未入库，这里下载。可设 PROXY=http://127.0.0.1:7897。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$ROOT/third_party"; mkdir -p "$TP"
PXOPT=""; [ -n "${PROXY:-}" ] && PXOPT="-x $PROXY"

if [ ! -f "$TP/nanoflann.hpp" ]; then
  echo "==> nanoflann.hpp"
  curl $PXOPT -sL -o "$TP/nanoflann.hpp" \
    https://raw.githubusercontent.com/jlblancoc/nanoflann/master/include/nanoflann.hpp
fi
if [ ! -d "$TP/Eigen" ]; then
  echo "==> Eigen 3.4.0"
  curl $PXOPT -sL -o /tmp/eigen.tar.gz \
    https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  tar -xzf /tmp/eigen.tar.gz -C /tmp
  cp -r /tmp/eigen-3.4.0/Eigen "$TP/Eigen"
fi
echo "==> deps ready in third_party/ (或改用系统 apt install libeigen3-dev)"
