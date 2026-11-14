#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../.."
RUN=$1
PY=${PY:-python}
until [ -f "$RUN/selection.json" ]; do sleep 60; done
EP=$($PY -c "import json;print(json.load(open('$RUN/selection.json'))['best_epoch'])")
if [ "$EP" = "0" ]; then echo "# 验证集上官方权重最好(best_epoch 0),仍用最后一轮评测作对照"; EP=$(ls $RUN/epoch*.pth | sed 's/.*epoch//; s/\.pth//' | sort -n | tail -1); fi
CK=$RUN/epoch$EP.pth
echo "# $(date '+%F %T') checkpoint $CK"
stem() { echo "$1" | sed -E 's/BusinessCampus/business_campus/; s/OldTown/old_town/' | tr A-Z a-z; }
for s in CustomCampus RobotCar BusinessCampus Neighborhood OldTown NCLT Boreas RobotCarCa4050 RobotCarCa5060 RobotCarCa7080 RobotCarCfull MSLS MSLSCa4050 MSLSCa5060 MSLSCa7080 MSLSCfull Pitts30k Pitts30kCa4050 Pitts30kCa5060 Pitts30kCa7080 Pitts30kCfull; do
  case $s in Pitts30k*|MSLS*) TH="--success-thresholds 10,15,25";; *) TH="";; esac
  echo "# $(date '+%F %T') start $s"
  if HF_HUB_OFFLINE=1 DATASET=$(stem $s) $PY tools/run_pairvpr_s2_rerank.py --scenes $s --pool MegaLoc_8448 $TH --checkpoint $CK --method PairVPR-s2-target; then
    echo "# $(date '+%F %T') ok $s"
  else
    echo "# $(date '+%F %T') FAIL $s"
  fi
done
echo "# $(date '+%F %T') ALL DONE"
