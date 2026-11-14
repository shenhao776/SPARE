#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../.."
CK=$1
PY=${PY:-python}
[ -f "$CK" ] || { echo "missing checkpoint: $CK"; exit 1; }
echo "# $(date '+%F %T') checkpoint $CK"
stem() { echo "$1" | sed -E 's/BusinessCampus/business_campus/; s/OldTown/old_town/' | tr A-Z a-z; }
for s in ${SCENES:-RobotCar BusinessCampus Neighborhood OldTown NCLT Boreas CustomCampus Pitts30k MSLS Pitts30kCa4050 Pitts30kCa5060 Pitts30kCa7080 Pitts30kCfull MSLSCa4050 MSLSCa5060 MSLSCa7080 MSLSCfull}; do
  case $s in Pitts30k*|MSLS*) TH="--success-thresholds 10,15,25";; *) TH="";; esac
  echo "# $(date '+%F %T') start $s"
  if DATASET=$(stem $s) $PY tools/run_selavpr_rerank.py --scenes $s --pool MegaLoc_8448 $TH --weights "$CK" --method SelaVPR-target; then
    echo "# $(date '+%F %T') ok $s"
  else
    echo "# $(date '+%F %T') FAIL $s"
  fi
done
echo "# $(date '+%F %T') ALL DONE"
