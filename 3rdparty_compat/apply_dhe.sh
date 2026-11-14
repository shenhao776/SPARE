#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/3rdparty_compat/DHE-VPR"
DST="$ROOT/3rdparty/DHE-VPR"
[ -d "$DST" ] || { echo "找不到 $DST —— 先 git clone https://github.com/Lu-Feng/DHE-VPR.git"; exit 1; }
for f in network.py dataset_geoloc.py datasets_ws.py dataset_qp.py util.py train_dhe.py datasets_util.py; do
  cp -v "$SRC/$f" "$DST/$f"
done
echo "已装到 $DST(改动说明见 $SRC/COMPAT_NOTES.md)"
