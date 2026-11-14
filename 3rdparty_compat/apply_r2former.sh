#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/3rdparty_compat/R2Former"
DST="$ROOT/3rdparty/R2Former"
[ -d "$DST" ] || { echo "找不到 $DST —— 先 git clone https://github.com/bytedance/R2Former.git"; exit 1; }
for f in util.py datasets_ws.py; do
  cp -v "$SRC/$f" "$DST/$f"
done
mkdir -p "$DST/model"
for f in model/network.py; do
  cp -v "$SRC/$f" "$DST/$f"
done
echo "已装到 $DST(改动说明见 $SRC/COMPAT_NOTES.md)"
