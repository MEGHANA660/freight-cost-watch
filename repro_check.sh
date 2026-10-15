#!/usr/bin/env bash
# Three independent runs (separate processes, separate output dirs); every file must be byte-identical.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf outputs/repro && mkdir -p outputs/repro
for i in 1 2 3; do python3 run.py --out "outputs/repro/run$i" > /dev/null; done
for f in output.csv weekly_metrics.csv cost_log.json; do
  for i in 2 3; do diff -q "outputs/repro/run1/$f" "outputs/repro/run$i/$f"; done
done
sha256sum outputs/repro/run*/output.csv
echo "REPRODUCIBLE: 3 runs byte-identical (output.csv, weekly_metrics.csv, cost_log.json)"
