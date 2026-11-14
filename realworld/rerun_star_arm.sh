#!/usr/bin/env bash
set -euo pipefail
WEIGHT=${1:?重排权重 .pth}
OUT=${2:?输出目录}
OLD=${3:-realworld/reloc_results}
MAP=${MAP:-$HOME/datasets/my_map_data/PCD/map_final_visualization.pcd}
RELOC_SEED=${RELOC_SEED:-$HOME/archive/2026-08_reloc_ndt_cpp/reloc_seed}
PY=${PY:-$HOME/miniconda3/envs/vpr_env/bin/python}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-$HOME/miniconda3/envs/vpr_env/lib}
NICE=${NICE:-10}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
MIN_FREE_MB=${MIN_FREE_MB:-3500}
FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "$FREE_MB" ] && [ "$FREE_MB" -lt "$MIN_FREE_MB" ]; then
  echo "显存空闲 ${FREE_MB} MB < ${MIN_FREE_MB} MB:用户的 GPU 任务还在,先别跑(MIN_FREE_MB 可调)"; exit 2
fi
echo "# nice $NICE, OMP $OMP_NUM_THREADS 线程, 空闲显存 ${FREE_MB:-?} MB, load $(cut -d' ' -f1-3 /proc/loadavg)"

[ -f "$WEIGHT" ] || { echo "权重不存在: $WEIGHT"; exit 1; }
[ -f "$MAP" ] || { echo "地图不存在: $MAP"; exit 1; }
[ -x "$RELOC_SEED" ] || { echo "reloc_seed 不可执行: $RELOC_SEED"; exit 1; }
[ -f "$OLD/star_poses.json" ] && [ -f "$OLD/ndt.csv" ] || { echo "旧凭据不全: $OLD"; exit 1; }
mkdir -p "$OUT/work"

echo "# 1/4 run_reloc_trials(raw + star,新 head)"
if [ ! -f "$OUT/star_poses.json" ]; then
  STAR_RERANK_WEIGHT="$PWD/$WEIGHT" DATASET=customcampus nice -n "$NICE" "$PY" realworld/run_reloc_trials.py --out "$OUT" || true
  [ -f "$OUT/star_poses.json" ] || { echo "run_reloc_trials 没有产出 $OUT/star_poses.json"; exit 1; }
fi

echo "# 2/4 把旧凭据的 fol 臂拷进来(与 head 无关)"
"$PY" - "$OLD/star_poses.json" "$OUT/star_poses.json" <<'EOF'
import json, sys
old = {r["trial"]: r for r in json.load(open(sys.argv[1], encoding="utf-8"))}
rows = json.load(open(sys.argv[2], encoding="utf-8"))
n = 0
for r in rows:
    o = old.get(r["trial"])
    if o and o.get("fol") and not r.get("fol"):
        r["fol"] = o["fol"]; n += 1
    # raw 臂与 head 无关,应逐 trial 相同;不同说明库特征或检索器变了,必须停下
    if o and o.get("raw") and r.get("raw") and o["raw"].get("pose7") != r["raw"].get("pose7"):
        raise SystemExit(f"trial {r['trial']}: raw 臂初值与旧凭据不同,检索侧变了,不能只重跑 star 臂")
json.dump(rows, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"  fol 臂拷入 {n}/{len(rows)}")
EOF

echo "# 3/4 作业表(只 star:60)+ NDT 配准"
"$PY" realworld/make_reloc_jobs.py --poses "$OUT/star_poses.json" --out "$OUT/jobs.txt" --arms star:60
STAR_CSV="$OUT/work/star60_ndt.csv"
if [ ! -f "$STAR_CSV" ] || [ "$(wc -l < "$STAR_CSV")" -lt 2 ]; then
  env LD_LIBRARY_PATH="${RELOC_LD_LIBRARY_PATH:-}" nice -n "$NICE" "$RELOC_SEED" "$MAP" realworld/reloc_frames "$OUT/jobs.txt" "$STAR_CSV"
fi

echo "# 4/4 合并 raw60 / fol60(旧)+ star60(新)"
{ head -1 "$OLD/ndt.csv"; grep -E '^[0-9]+,(raw60|fol60),' "$OLD/ndt.csv"; tail -n +2 "$STAR_CSV"; } > "$OUT/ndt.csv"
echo "  $(wc -l < "$OUT/ndt.csv") 行(含表头) -> $OUT/ndt.csv"
echo "下一步:$PY realworld/summarize_reloc.py --dir $OUT --csv results/tables/robot_trials.csv --paired-csv results/tables/robot_trials_paired.csv"
