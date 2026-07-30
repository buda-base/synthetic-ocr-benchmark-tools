#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

PROJECT=/home/eroux/BUDA/softs/synthetic-ocr-benchmark-tools
PYTHON=/opt/synthbench-venv/bin/python
PLAN="$PROJECT/local_review_plan_500_v4.parquet"
OUT=/mnt/synthbench/local_review_500_v4
RESULT_URI=s3://bec.bdrc.io/synthetic/reviews/local_review_500_v4/
LOG=/mnt/synthbench/synthbench-review-500-v4.log

exec > >(tee -a "$LOG") 2>&1
cd "$PROJECT"
mkdir -p "$OUT"

if [[ -s "$PLAN" ]]; then
  echo "[plan] Reusing existing balanced review plan: $PLAN"
else
  echo "[plan] Building a balanced 250-Uchen / 250-Ume review plan"
  "$PYTHON" synthetic_benchmark/build_balanced_review_plan.py \
    --output "$PLAN" \
    --target-images 500 \
    --seed 13
fi

echo "[render] Starting balanced 500-page review generation"
"$PYTHON" synthetic_benchmark/render_batches.py \
  "$PLAN" \
  --out-dir "$OUT" \
  --document-augmentation \
  --enable-shorthands \
  --font-augmentation-variants 12 \
  --font-augmentation-workers 8 \
  --font-augmentation-s3-cache-uri s3://bec.bdrc.io/synthetic/font_cache/ \
  --font-augmentation-local-cache-dir /mnt/synthbench/font_cache \
  --batch-size 24 \
  --jobs 8 \
  --force

echo "[review] Creating flat review folder"
"$PYTHON" synthetic_benchmark/create_flat_review_folder.py "$OUT"

echo "[upload] Uploading review images and metadata"
aws s3 sync "$OUT/flat/" "$RESULT_URI" --only-show-errors
for manifest in \
  document_augmentation_manifest.json \
  font_augmentation_manifest.json \
  image_output_manifest.json \
  page_layout_manifest.json \
  raster_naturalization_manifest.json
do
  aws s3 cp "$OUT/$manifest" \
    "${RESULT_URI}metadata/$manifest" --only-show-errors
done
for report in \
  "${PLAN%.parquet}.stack_coverage_by_font.csv" \
  "${PLAN%.parquet}.stack_rarity_by_font.csv" \
  "${PLAN%.parquet}.stack_coverage_summary.json"
do
  aws s3 cp "$report" \
    "${RESULT_URI}metadata/$(basename "$report")" --only-show-errors
done
printf 'instance_id=%s\ncompleted_at=%s\n' \
  "$(hostname)" \
  "$(date --iso-8601=seconds)" |
  aws s3 cp - "${RESULT_URI}_SUCCESS" --only-show-errors

echo "[done] Balanced review generation complete: $RESULT_URI"
