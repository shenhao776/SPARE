#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
OUT=${1:-assets/timing_$(date +%Y%m%d)}
mkdir -p "$OUT"
LOG="$OUT/timing.log"
W=assets/MegaLoc_8448/models/vpr_reranking_gru_arch=single_frame_deriv=False_prog=False_area=True_dino=True_zoom=True_vprpatch=True_innorm=False_UnionMaps_no_roi.pth
gpu() { nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; }
step() { echo "# $(date '+%F %T') $* | GPU $(gpu)" | tee -a "$LOG"; }

step "SPARE (universal head, --warm-db)"
STAR_RERANK_WEIGHT=$W DATASET=robotcar $PY tools/bench_online.py --scene RobotCar --n 50 --warm-db \
    2>&1 | grep -E "GPU:|对数|ms" | tee -a "$LOG"

for s in vop r2former fol selavpr pairvpr_s2 geowarp; do
    step "$s"
    HF_HUB_OFFLINE=1 DATASET=robotcar $PY tools/run_${s}_rerank.py --scenes RobotCar \
        --pool MegaLoc_8448 --output-dir "$OUT" --force 2>&1 | tail -2 | tee -a "$LOG"
done

step "RoMa (纯匹配)"
DATASET=robotcar $PY tools/run_roma_gv_rerank.py --scenes RobotCar --n-queries 21 \
    --bench-json "$OUT/roma_bench.json" 2>&1 | tail -2 | tee -a "$LOG"
step "DONE"
