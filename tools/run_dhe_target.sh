#!/usr/bin/env bash
set -euo pipefail

SCENE="${1:?用法: run_dhe_target.sh <场景> <阈值米数> [--smoke]}"
TH="${2:?用法: run_dhe_target.sh <场景> <阈值米数> [--smoke]}"
SMOKE=0
EPOCHS=""
case "${3:-}" in
  --smoke) SMOKE=1 ;;
  --epochs) EPOCHS="${4:?--epochs 后要跟轮数}" ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=${PY:-python}
DATA="$ROOT/assets/prgs_data"
FE="$ROOT/assets/dhe_vpr/CCT14_msls.pth"

[ -d "$DATA/$SCENE" ] || { echo "没有导出数据: $DATA/$SCENE(跑 tools/export_prgs_data.py)" >&2; exit 1; }
[ -f "$FE" ] || { echo "缺 $FE —— 见 3rdparty_compat/DHE-VPR/COMPAT_NOTES.md 的下载命令" >&2; exit 1; }

BATCH=4
WORKERS=2

EXTRA=""
TAG="${SCENE}_th${TH}"
if [ "$SMOKE" = "1" ]; then
  EXTRA="--n_epochs 1 --iterations_per_epoch 50"
  TAG="${TAG}_smoke"
elif [ -n "$EPOCHS" ]; then
  EXTRA="--n_epochs $EPOCHS"
fi
LOG="$ROOT/logs/dhe/${TAG}.log"
PIDF="$ROOT/logs/dhe/${TAG}.pid"
mkdir -p "$ROOT/logs/dhe"

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
export MKL_THREADING_LAYER=GNU
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GC_FRAC=$(awk -v t="$TOTAL_MIB" 'BEGIN{f=(t-2500)/t; if(f<0.5)f=0.5; printf "%.2f", f}')
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:${GC_FRAC}"
echo "[显存] 卡 ${TOTAL_MIB}MiB,回收阈值 ${GC_FRAC}"

cd "$ROOT/3rdparty/DHE-VPR"

setsid nohup bash -c '
  trap "echo; echo \"[中断] 收到信号,训练未完成 —— 不归档\"; exit 130" INT TERM
  stdbuf -oL -eL "$1" train_dhe.py \
    --datasets_folder "$2" --dataset_name "$3" \
    --resume_fe "$4" \
    --exp_name "star_$5" \
    --batch_size "$6" --num_workers "$7" \
    --positive_dist_threshold "$8" ${10}
  rc=$?
  echo "[训练退出码] $rc"
  if [ $rc -eq 0 ]; then
    "$1" "$9/tools/archive_dhe_weight.py" --tag "$5" \
      || echo "[归档] 失败(训练本身已成功,可手动补:python tools/archive_dhe_weight.py --tag $5)"
  else
    echo "[归档] 跳过 —— 训练非正常退出"
  fi
' _ "$PY" "$DATA" "$SCENE" "$FE" "$TAG" "$BATCH" "$WORKERS" "$TH" "$ROOT" "$EXTRA" \
  > "$LOG" 2>&1 < /dev/null &

echo $! > "$PIDF"
sleep 2
echo "[起] DHE $TAG pid=$(cat "$PIDF")  日志=$LOG"
