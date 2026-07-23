#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

PROJECT=/home/eroux/BUDA/softs/synthetic-ocr-benchmark-tools
PYTHON=/opt/synthbench-venv/bin/python
PLAN="$PROJECT/local_review_plan_1000_v2.parquet"
OUT=/mnt/synthbench/local_review_1000_v2
RESULT_URI=s3://bec.bdrc.io/synthetic/reviews/local_review_1000_v2/

cd "$PROJECT"
mkdir -p "$OUT"

echo "[plan] Building a 500-Uchen / 500-Ume plan with every eligible font face"
"$PYTHON" synthetic_benchmark/build_balanced_review_plan.py \
  --output "$PLAN" \
  --target-images 1000 \
  --seed 13

echo "[render] Starting balanced 1,000-page review generation"
"$PYTHON" synthetic_benchmark/render_batches.py \
  "$PLAN" \
  --out-dir "$OUT" \
  --document-augmentation \
  --enable-shorthands \
  --font-augmentation-variants 12 \
  --font-augmentation-workers 8 \
  --font-augmentation-s3-cache-uri s3://bec.bdrc.io/synthetic/font_cache/ \
  --batch-size 24 \
  --jobs 8 \
  --force

echo "[review] Creating flat review folder"
"$PYTHON" synthetic_benchmark/create_flat_review_folder.py "$OUT"

echo "[upload] Uploading to a clean, versioned review prefix"
aws s3 sync "$OUT/flat/" "$RESULT_URI" --only-show-errors
for manifest in \
  document_augmentation_manifest.json \
  font_augmentation_manifest.json \
  image_output_manifest.json \
  page_layout_manifest.json
do
  aws s3 cp "$OUT/$manifest" \
    "${RESULT_URI}metadata/$manifest" --only-show-errors
done
printf 'instance_id=%s\ncompleted_at=%s\n' \
  "$(hostname)" \
  "$(date --iso-8601=seconds)" |
  aws s3 cp - "${RESULT_URI}_SUCCESS" --only-show-errors

echo "[done] Balanced review generation complete: $RESULT_URI"
