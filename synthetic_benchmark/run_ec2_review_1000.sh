#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1

PROJECT=/home/eroux/BUDA/softs/synthetic-ocr-benchmark-tools
PAYLOAD_URI=s3://bec.bdrc.io/synthetic/staging/local_review_1000_payload.tar.zst
RESULT_URI=s3://bec.bdrc.io/synthetic/reviews/local_review_1000/
LOG=/var/log/synthbench-review-1000.log

exec > >(tee -a "$LOG") 2>&1

echo "[bootstrap] Installing system dependencies"
apt-get update
apt-get install -y --no-install-recommends \
  awscli fontforge-nox libgl1 libglib2.0-0 libharfbuzz-bin \
  poppler-utils python3-pip python3-venv texlive-fonts-recommended \
  texlive-latex-extra texlive-luatex zstd

echo "[bootstrap] Preparing instance-store workspace"
DEVICE=$(
  lsblk -dpno NAME,MODEL |
    awk '/Amazon EC2 NVMe Instance Storage/ {print $1; exit}'
)
if [[ -n "$DEVICE" ]]; then
  mkfs.ext4 -F "$DEVICE"
  mkdir -p /mnt/synthbench
  mount "$DEVICE" /mnt/synthbench
else
  mkdir -p /mnt/synthbench
fi

echo "[bootstrap] Downloading generation payload"
mkdir -p "$PROJECT"
aws s3 cp "$PAYLOAD_URI" /tmp/local_review_1000_payload.tar.zst
tar --zstd -xf /tmp/local_review_1000_payload.tar.zst -C "$PROJECT"

echo "[bootstrap] Installing Python environment"
python3 -m venv /opt/synthbench-venv
/opt/synthbench-venv/bin/pip install --upgrade pip
/opt/synthbench-venv/bin/pip install \
  -r "$PROJECT/synthetic_benchmark/requirements-render.txt"

OUT=/mnt/synthbench/local_review_1000
echo "[render] Starting 1,000-page review generation"
cd "$PROJECT"
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

echo "[review] Creating zero-copy flat review folder"
/opt/synthbench-venv/bin/python \
  synthetic_benchmark/create_flat_review_folder.py "$OUT"

echo "[upload] Uploading flat review and metadata"
aws s3 sync "$OUT/flat/" "$RESULT_URI" --only-show-errors
aws s3 cp "$OUT/document_augmentation_manifest.json" \
  "${RESULT_URI}metadata/document_augmentation_manifest.json" --only-show-errors
aws s3 cp "$OUT/font_augmentation_manifest.json" \
  "${RESULT_URI}metadata/font_augmentation_manifest.json" --only-show-errors
aws s3 cp "$OUT/image_output_manifest.json" \
  "${RESULT_URI}metadata/image_output_manifest.json" --only-show-errors
aws s3 cp "$OUT/page_layout_manifest.json" \
  "${RESULT_URI}metadata/page_layout_manifest.json" --only-show-errors
printf 'instance_id=%s\ncompleted_at=%s\n' \
  "$(hostname)" \
  "$(date --iso-8601=seconds)" |
  aws s3 cp - "${RESULT_URI}_SUCCESS" --only-show-errors

echo "[done] Review generation complete: $RESULT_URI"
