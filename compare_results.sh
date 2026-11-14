#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

TOL=${COMPARE_TOL:-0.1}
fail=0
near_tables=0

for f in main.csv backbones.csv ablation.csv threshold25.csv crop_sweep.csv crop_sweep_pitts30k.csv crop_sweep_msls.csv; do
    if [ ! -f "results/replay/$f" ]; then
        case "$f" in
            crop_sweep*.csv)
                echo "skip  $f (not written by the table replay; checked by extra_reproduce.py)"
                continue ;;
        esac
        echo "MISSING  results/replay/$f -- run 'bash full_reproduce.sh' first"
        fail=1; continue
    fi
    if cmp -s "results/replay/$f" "results/tables/$f"; then
        echo "ok    $f (byte-identical to the published numbers)"
        continue
    fi
    verdict=$(awk -v tol="$TOL" '
        function nums(s, arr) { gsub(/[^0-9.]+/, " ", s); return split(s, arr, " ") }
        function shape(s)     { gsub(/[0-9]+(\.[0-9]+)?/, "#", s); return s }
        NR == FNR { pub[FNR] = $0; npub = FNR; next }
        {
            if (FNR > npub)            { structural = 1; next }
            if (pub[FNR] == $0)        next
            if (shape(pub[FNR]) != shape($0)) { structural = 1; next }
            np = nums(pub[FNR], A); nr = nums($0, B)
            if (np != nr)              { structural = 1; next }
            for (i = 1; i <= np; i++) {
                d = A[i] - B[i]; if (d < 0) d = -d
                if (d > 1e-9) {
                    cells++
                    if (d > worst) worst = d
                    if (d > tol + 1e-9) over++
                }
            }
        }
        END {
            if (FNR < npub) structural = 1
            if (structural) { print "STRUCT"; exit }
            printf "%s %d %.2f\n", (over ? "OVER" : "NEAR"), cells, worst
        }' "results/tables/$f" "results/replay/$f")

    case "$verdict" in
        NEAR*)
            set -- $verdict
            echo "≈     $f ($2 cell(s) differ, at most $3 -- within the $TOL tolerance)"
            near_tables=$((near_tables + 1))
            ;;
        *)
            echo "FAIL  $f"
            diff "results/tables/$f" "results/replay/$f" | head -10 || true
            fail=1
            ;;
    esac
done

if [ "$fail" = 0 ]; then
    if [ "$near_tables" = 0 ]; then
        echo "PASS -- the replay reproduced the paper cell for cell."
    else
        echo "PASS -- the replay reproduced the paper; $near_tables table(s) agree to within $TOL"
        echo "        (marked ≈; expected on a CPU-only replay, see README)."
    fi
else
    echo "FAIL -- differences above exceed the $TOL tolerance."
fi
exit $fail
