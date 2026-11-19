#!/bin/bash
# Run ON LS6 login after overnight jobs finish, then rsync from the Mac.
#   bash experiments/mopd/runs/pull_overnight_results.sh
set -euo pipefail
cd "${HOPD_HOME:-$SCRATCH/hopd/H-OPD}"
# shellcheck disable=SC1090
source "${HOPD_VENV:-$SCRATCH/hopd/.venv}/bin/activate"

JOBS=(3593543 3593544 3593882)

echo "======== sacct ========"
sacct -j "$(IFS=,; echo "${JOBS[*]}")" --format=JobID,JobName,State,Elapsed,ExitCode,End,NodeList -P

for j in "${JOBS[@]}"; do
  echo ""
  echo "======== mopd_$j ========"
  RUN="logs/mopd_$j"
  OUTDIR="experiments/mopd/runs/mopd_$j"
  mkdir -p "$OUTDIR/cluster"
  if [[ ! -d "$RUN" ]]; then
    echo "MISSING $RUN"
    continue
  fi
  python3 experiments/mopd/watch.py --run "$RUN" --last 999 \
    > "$OUTDIR/cluster/watch_table.txt" || true
  python3 experiments/mopd/watch.py --run "$RUN" \
    --csv "$OUTDIR/cluster/metrics.csv" || true
  cp -f "$RUN/metrics.jsonl" "$OUTDIR/cluster/" 2>/dev/null || true
  # job .out (name may be hopd-p20k_ / hopd-r256_ / hopd-p55k_)
  shopt -s nullglob
  for f in logs/hopd-*_${j}.out; do
    cp -f "$f" "$OUTDIR/cluster/"
    echo "copied $f"
    # show last failure / Done / val
    echo "--- tail $f ---"
    tail -n 30 "$f" || true
  done
  shopt -u nullglob
  echo "--- watch last rows ---"
  tail -n 25 "$OUTDIR/cluster/watch_table.txt" || true
done

echo ""
echo "Done on cluster. On Mac:"
echo "  rsync -avP ls6:/scratch/11289/rkr758/hopd/H-OPD/experiments/mopd/runs/mopd_3593543/ \\"
echo "    /Users/admin/Projects/H-OPD/experiments/mopd/runs/mopd_3593543/"
echo "  (repeat for 3593544 and 3593882)"
echo "  Also pull rollouts/val:"
echo "  rsync -avP ls6:/scratch/11289/rkr758/hopd/H-OPD/logs/mopd_JOBID/rollouts/ ..."
echo "  rsync -avP ls6:/scratch/11289/rkr758/hopd/H-OPD/logs/mopd_JOBID/val/ ..."
