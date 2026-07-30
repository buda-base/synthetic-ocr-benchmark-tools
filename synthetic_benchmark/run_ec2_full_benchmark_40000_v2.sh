#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
# Augraphy's Folding helper currently cannot JIT-compile under Python 3.13.
# Its NumPy implementation is fast enough and avoids the Numba dispatcher crash.
export NUMBA_DISABLE_JIT=1

PROJECT=/home/eroux/BUDA/softs/synthetic-ocr-benchmark-tools
PYTHON=/opt/synthbench-venv/bin/python
PLAN="$PROJECT/full_benchmark_plan_40000_category_balanced_v2.parquet"
OUT=/mnt/synthbench/full_benchmark_40000_v2
LOG=/mnt/synthbench/full-benchmark-40000-v2.log

MW_ID=MW1BCS002
W_ID=W1BCS002
IE_ID=IE1BCS002
I_ID_PREFIX=I1BCS002
VE_ID_PREFIX=VE1BCS002
VOLUME_VERSION=v001
BENCHMARK_VERSION=202604
DATASET_ID=BECSynthetic_02

IMAGE_URI="s3://bec.bdrc.io/ocr_benchmark/images/$W_ID/"
ALIGNMENT_URI="s3://bec.bdrc.io/ocr_benchmark/alignments/$BENCHMARK_VERSION/$DATASET_ID/"
RUN_URI=s3://bec.bdrc.io/synthetic/benchmarks/full_40000_v2/

exec > >(tee -a "$LOG") 2>&1
cd "$PROJECT"
mkdir -p "$OUT"

if [[ -s "$PLAN" ]]; then
  echo "[plan] Reusing existing 40,000-page plan: $PLAN"
else
  echo "[plan] Building a plan balanced across seven populated script_8 categories"
  "$PYTHON" synthetic_benchmark/build_render_plan.py \
    --output "$PLAN" \
    --support-parquet "$PROJECT/coverage_report/out/stack_support.parquet" \
    --target-images 40000 \
    --top-up-category-quotas \
    --seed 13
fi

echo "[plan] Verifying production category balance"
"$PYTHON" - "$PLAN" <<'PY'
import collections
import sys

import pyarrow.parquet as pq

rows = pq.read_table(sys.argv[1], columns=["script_8"]).to_pylist()
counts = collections.Counter(str(row["script_8"]) for row in rows)
if len(rows) != 40_000:
    raise SystemExit(f"Expected 40,000 plan rows, found {len(rows)}")
if len(counts) != 7:
    raise SystemExit(f"Expected seven populated script_8 categories, found {counts}")
if max(counts.values()) - min(counts.values()) > 1:
    raise SystemExit(f"Production categories are not balanced: {counts}")
print(f"[plan] Verified script_8 counts: {dict(sorted(counts.items()))}")
PY

periodic_image_upload() {
  while true; do
    if [[ -d "$OUT/images/$W_ID" ]]; then
      echo "[upload] Periodic image synchronization"
      aws s3 sync "$OUT/images/$W_ID/" "$IMAGE_URI" --only-show-errors
    fi
    sleep 300
  done
}

periodic_image_upload &
UPLOAD_PID=$!
cleanup_uploader() {
  kill "$UPLOAD_PID" 2>/dev/null || true
  wait "$UPLOAD_PID" 2>/dev/null || true
}
trap cleanup_uploader EXIT

echo "[render] Starting 40,000-page production benchmark"
"$PYTHON" synthetic_benchmark/render_batches.py \
  "$PLAN" \
  --out-dir "$OUT" \
  --mw-id "$MW_ID" \
  --w-id "$W_ID" \
  --ie-id "$IE_ID" \
  --i-id-prefix "$I_ID_PREFIX" \
  --ve-id-prefix "$VE_ID_PREFIX" \
  --volume-version "$VOLUME_VERSION" \
  --benchmark-version "$BENCHMARK_VERSION" \
  --dataset-id "$DATASET_ID" \
  --document-augmentation \
  --enable-shorthands \
  --font-augmentation-variants 12 \
  --font-augmentation-workers 8 \
  --font-augmentation-s3-cache-uri s3://bec.bdrc.io/synthetic/font_cache/ \
  --font-augmentation-local-cache-dir /mnt/synthbench/font_cache \
  --batch-size 48 \
  --jobs 12

echo "[upload] Final image synchronization"
aws s3 sync "$OUT/images/$W_ID/" "$IMAGE_URI" --only-show-errors

echo "[upload] Uploading alignments"
aws s3 sync \
  "$OUT/alignments/$BENCHMARK_VERSION/$DATASET_ID/" \
  "$ALIGNMENT_URI" \
  --only-show-errors
aws s3 cp \
  "$OUT/alignments/$BENCHMARK_VERSION/datasets.csv" \
  "s3://bec.bdrc.io/ocr_benchmark/alignments/$BENCHMARK_VERSION/datasets.csv" \
  --only-show-errors

echo "[upload] Uploading run metadata"
shopt -s nullglob
for metadata in "$OUT"/*.json "$OUT"/*.csv; do
  aws s3 cp "$metadata" \
    "${RUN_URI}metadata/$(basename "$metadata")" --only-show-errors
done
for report in \
  "$PLAN" \
  "${PLAN%.parquet}.summary.csv" \
  "${PLAN%.parquet}.stack_coverage_by_font.csv" \
  "${PLAN%.parquet}.stack_rarity_by_font.csv" \
  "${PLAN%.parquet}.stack_coverage_summary.json"
do
  if [[ -f "$report" ]]; then
    aws s3 cp "$report" \
      "${RUN_URI}metadata/$(basename "$report")" --only-show-errors
  fi
done

PHYSICAL_IMAGES="$("$PYTHON" - "$OUT/alignments/$BENCHMARK_VERSION/datasets.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], encoding="utf-8", newline="") as stream:
    print(next(csv.DictReader(stream))["nb_img_files"])
PY
)"
printf 'instance_id=%s\ncompleted_at=%s\nplanned_rows=40000\nphysical_images=%s\nw_id=%s\nie_id=%s\ndataset_id=%s\n' \
  "$(hostname)" \
  "$(date --iso-8601=seconds)" \
  "$PHYSICAL_IMAGES" \
  "$W_ID" \
  "$IE_ID" \
  "$DATASET_ID" |
  aws s3 cp - "${RUN_URI}_SUCCESS" --only-show-errors

cleanup_uploader
trap - EXIT
echo "[done] Full benchmark complete: $RUN_URI"
