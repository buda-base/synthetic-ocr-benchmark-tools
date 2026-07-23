#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
PROJECT=/home/eroux/BUDA/softs/synthetic-ocr-benchmark-tools
OUT=/mnt/synthbench/local_review_1000
RESULT_URI=s3://bec.bdrc.io/synthetic/reviews/local_review_1000/
LOG=/var/log/synthbench-review-1000.log

exec > >(tee -a "$LOG") 2>&1
cd "$PROJECT"

/opt/synthbench-venv/bin/pip install \
  -r synthetic_benchmark/requirements-render.txt

/opt/synthbench-venv/bin/python synthetic_benchmark/render_batches.py \
  local_review_plan_1000.parquet \
  --out-dir "$OUT" \
  --document-augmentation \
  --enable-shorthands \
  --font-augmentation-variants 12 \
  --font-augmentation-workers 8 \
  --font-augmentation-s3-cache-uri s3://bec.bdrc.io/synthetic/font_cache/ \
  --batch-size 24 \
  --jobs 8 \
  --force

/opt/synthbench-venv/bin/python \
  synthetic_benchmark/create_flat_review_folder.py "$OUT"
aws s3 sync "$OUT/flat/" "$RESULT_URI" --only-show-errors
for manifest in \
  document_augmentation_manifest.json font_augmentation_manifest.json \
  image_output_manifest.json page_layout_manifest.json; do
  aws s3 cp "$OUT/$manifest" "${RESULT_URI}metadata/$manifest" \
    --only-show-errors
done
printf 'instance=%s\ncompleted_at=%s\n' "$(hostname)" "$(date --iso-8601=seconds)" |
  aws s3 cp - "${RESULT_URI}_SUCCESS" --only-show-errors
echo "[done] Review generation complete: $RESULT_URI"
