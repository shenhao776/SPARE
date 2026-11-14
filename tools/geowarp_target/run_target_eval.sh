#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../.."
RUN=$1
PY=${PY:-python}
OUT=assets/geowarp_target
$PY tools/geowarp_target/select_epoch.py --run "$RUN" --every 10 --merge --out $OUT/selection.json || exit 1
HR=$OUT/homography_regression_best.torch
if [ ! -f "$HR" ]; then
  echo "# released weights are best on validation; evaluating the last epoch as the fine-tuned row"
  HR=$(ls "$RUN"/checkpoints/homography_regression_*.torch | sort | tail -1)
fi
echo "# $(date '+%F %T') homography regression $HR"
stem() { echo "$1" | sed -E 's/BusinessCampus/business_campus/; s/OldTown/old_town/' | tr A-Z a-z; }
for s in ${SCENES:-RobotCar BusinessCampus Neighborhood OldTown NCLT Boreas CustomCampus Pitts30k MSLS Pitts30kCa4050 Pitts30kCa5060 Pitts30kCa7080 Pitts30kCfull MSLSCa4050 MSLSCa5060 MSLSCa7080 MSLSCfull}; do
  case $s in Pitts30k*|MSLS*) TH="--success-thresholds 10,15,25";; *) TH="";; esac
  echo "# $(date '+%F %T') start $s"
  if DATASET=$(stem $s) $PY tools/run_geowarp_rerank.py --scenes $s --pool MegaLoc_8448 $TH --hr-weights "$HR" --method GeoWarp-target; then
    echo "# $(date '+%F %T') ok $s"
  else
    echo "# $(date '+%F %T') FAIL $s"
  fi
done
echo "# $(date '+%F %T') ALL DONE"
