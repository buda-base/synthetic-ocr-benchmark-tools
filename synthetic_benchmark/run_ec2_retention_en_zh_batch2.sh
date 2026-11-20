#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export NUMBA_DISABLE_JIT=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_NUM_THREADS=1

PROJECT=/mnt/synthbench/repo
PYTHON=/opt/synthbench-venv/bin/python
OUT=/mnt/synthbench/retention_en_zh_batch2
CACHE=/mnt/synthbench/retention_cache
PREVIOUS=/mnt/synthbench/retention_en_zh_batch1/render_manifest.parquet
LOG=/mnt/synthbench/retention-en-zh-batch2.log
PAYLOAD_URI=s3://bec.bdrc.io/synthetic/staging/retention_en_zh_batch2_payload.tar.zst
RUN_URI=s3://bec.bdrc.io/synthetic/benchmarks/retention_en_zh_batch2/
EVAL_URI=s3://bec.bdrc.io/evaluation_benchmark/eval_parquets/yigdzin2_dev/dev-retain.parquet
EVAL=/mnt/synthbench/dev-retain.parquet

mkdir -p /mnt/synthbench "$PROJECT" "$CACHE"
exec > >(tee -a "$LOG") 2>&1

finish() {
  status=$?
  set +e
  aws s3 cp "$LOG" "${RUN_URI}retention-en-zh-batch2.log" --only-show-errors
  if [[ $status -ne 0 ]]; then
    printf 'instance_id=%s\nfailed_at=%s\nexit_code=%s\n' \
      "$(hostname)" "$(date --iso-8601=seconds)" "$status" |
      aws s3 cp - "${RUN_URI}_FAILED" --only-show-errors
  fi
  shutdown -h now
  exit "$status"
}
trap finish EXIT

if [[ "${RESUME_AFTER_RENDER:-0}" != 1 ]]; then
echo "[bootstrap] Installing renderer and licensed font packages"
apt-get update
apt-get install -y --no-install-recommends \
  awscli fontconfig fonts-arphic-ukai fonts-arphic-uming fonts-dejavu-core \
  fonts-ebgaramond fonts-liberation2 fonts-noto-cjk \
  fonts-noto-cjk-extra fonts-noto-core fonts-wqy-microhei fonts-wqy-zenhei \
  libgl1 libglib2.0-0 libharfbuzz0b libfribidi0 python3-pip python3-venv zstd
fc-cache -f

aws s3 cp "$PAYLOAD_URI" /tmp/retention_en_zh_batch2_payload.tar.zst
tar --zstd -xf /tmp/retention_en_zh_batch2_payload.tar.zst -C "$PROJECT"

if [[ ! -x "$PYTHON" ]]; then
  python3 -m venv /opt/synthbench-venv
fi
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r "$PROJECT/synthetic_benchmark/requirements-render.txt"

echo "[inputs] Fetching evaluation text and Batch 1 manifest"
aws s3 cp "$EVAL_URI" "$EVAL" --only-show-errors
aws s3 cp \
  s3://bec.bdrc.io/synthetic/benchmarks/retention_en_zh_batch1/render_manifest.parquet \
  "$PREVIOUS" --only-show-errors

echo "[smoke] Rendering two continuation pages per language"
rm -rf /mnt/synthbench/retention_batch2_smoke
cd "$PROJECT"
PYTHONPATH="$PROJECT/synthetic_benchmark" "$PYTHON" \
  synthetic_benchmark/render_retention_en_zh.py \
  --out-dir /mnt/synthbench/retention_batch2_smoke \
  --eval-parquet "$EVAL" \
  --count-per-language 2 \
  --start-image-id 5001 \
  --train-only \
  --source-docs-per-variant 40 \
  --cache-dir "$CACHE" \
  --jobs 2
rm -rf /mnt/synthbench/retention_batch2_smoke

echo "[render] Rendering 8,000 train-only pages per language"
rm -rf "$OUT"
PYTHONPATH="$PROJECT/synthetic_benchmark" "$PYTHON" \
  synthetic_benchmark/render_retention_en_zh.py \
  --out-dir "$OUT" \
  --eval-parquet "$EVAL" \
  --count-per-language 8000 \
  --start-image-id 5001 \
  --train-only \
  --source-docs-per-variant 700 \
  --prefer-new-source-manifest "$PREVIOUS" \
  --cache-dir "$CACHE" \
  --jobs 12
fi

echo "[validate] Checking Batch 2 quotas, IDs, splits, bounds, and exclusivity"
"$PYTHON" - "$OUT" "$PREVIOUS" <<'PY'
import collections
import sys
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
previous = pq.read_table(sys.argv[2]).to_pylist()
rows = pq.read_table(root / "render_manifest.parquet").to_pylist()
if len(rows) != 16_000:
    raise SystemExit(f"Expected 16,000 manifest rows, found {len(rows)}")
for language in ("en", "zh"):
    group = [row for row in rows if row["language_group"] == language]
    if len(group) != 8_000:
        raise SystemExit(f"{language}: expected 8,000 rows, found {len(group)}")
    ids = sorted(int(row["image_id"]) for row in group)
    if ids != list(range(5001, 13001)):
        raise SystemExit(f"{language}: unexpected image ID range")
if {row["suggested_split"] for row in rows} != {"train"}:
    raise SystemExit("Batch 2 contains a non-train split")
regions = [row for row in rows if row["layout_class"] == "region"]
if len(regions) != 8_000:
    raise SystemExit(f"Expected 8,000 regions, found {len(regions)}")
if not all(20 <= int(row["character_count"]) <= 1500 for row in regions):
    raise SystemExit("Region character bounds failed")
if not all(1 <= int(row["line_count"]) <= 30 for row in regions):
    raise SystemExit("Region line bounds failed")
if not all(row["transcription"] == unicodedata.normalize("NFC", row["transcription"]) for row in rows):
    raise SystemExit("Non-NFC target found")
if not all(
    row["transcription"] == row["transcription"].rstrip()
    and all(not line.endswith(" ") for line in row["transcription"].split("\n"))
    for row in rows
):
    raise SystemExit("Invalid target whitespace")
prior_sources = {row["source_document_id"] for row in previous}
new_sources = {row["source_document_id"] for row in rows}
if prior_sources & new_sources:
    raise SystemExit(f"Batch 2 reused {len(prior_sources & new_sources)} Batch 1 source documents")
prior_holdout_fonts = {
    row["font_name"] for row in previous if row["suggested_split"] in {"val", "test"}
}
new_fonts = {row["font_name"] for row in rows}
if prior_holdout_fonts & new_fonts:
    raise SystemExit(f"Batch 2 reused holdout fonts: {sorted(prior_holdout_fonts & new_fonts)}")
zh = [row for row in rows if row["language_group"] == "zh"]
hant = sum(row["language"] == "zh-Hant" for row in zh)
if not 0.08 <= hant / len(zh) <= 0.12:
    raise SystemExit(f"Unexpected zh-Hant share: {hant}/{len(zh)}")
images = list((root / "images").glob("**/*.jpg")) + list((root / "images").glob("**/*.tif"))
if len(images) != 16_000:
    raise SystemExit(f"Expected 16,000 images, found {len(images)}")
volumes = {path.parent.parent.name for path in images}
expected_volumes = {
    *(f"I1BCS007_{i:04d}" for i in range(6, 14)),
    *(f"I1BCS008_{i:04d}" for i in range(6, 14)),
}
if volumes != expected_volumes:
    raise SystemExit(f"Unexpected volumes: {sorted(volumes)}")
print({
    "pages": len(rows),
    "languages": dict(collections.Counter(row["language"] for row in rows)),
    "new_source_documents": len(new_sources),
    "font_faces": len(new_fonts),
})
PY

echo "[catalog] Merging Batch 2 volumes with existing Batch 1 catalogs"
for dataset in BECSynthetic_RetainEN BECSynthetic_RetainZH; do
  remote="s3://bec.bdrc.io/ocr_benchmark/alignments/202604/$dataset"
  local="$OUT/alignments/202604/$dataset"
  aws s3 cp "$remote/catalog_alignments.csv" "/tmp/$dataset.catalog_alignments.old.csv" --only-show-errors
  aws s3 cp "$remote/catalog_volumes.csv" "/tmp/$dataset.catalog_volumes.old.csv" --only-show-errors
  "$PYTHON" - \
    "/tmp/$dataset.catalog_alignments.old.csv" "$local/catalog_alignments.csv" \
    "/tmp/$dataset.catalog_volumes.old.csv" "$local/catalog_volumes.csv" <<'PY'
import csv
import sys

def merge(old_path, new_path, keys):
    with open(old_path, encoding="utf-8", newline="") as stream:
        old_reader = csv.DictReader(stream)
        fieldnames = list(old_reader.fieldnames or [])
        rows = list(old_reader)
    with open(new_path, encoding="utf-8", newline="") as stream:
        additions = list(csv.DictReader(stream))
    merged = {tuple(row[key] for key in keys): row for row in rows}
    merged.update({tuple(row[key] for key in keys): row for row in additions})
    with open(new_path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged[key] for key in sorted(merged))

merge(sys.argv[1], sys.argv[2], ("i_id", "ve_id"))
merge(sys.argv[3], sys.argv[4], ("coll_id", "vol_id"))
PY
done

echo "[upload] Uploading new volumes, merged catalogs, and reports"
for w_id in W1BCS007 W1BCS008; do
  aws s3 sync "$OUT/images/$w_id/" \
    "s3://bec.bdrc.io/ocr_benchmark/images/$w_id/" --only-show-errors
done
for dataset in BECSynthetic_RetainEN BECSynthetic_RetainZH; do
  local="$OUT/alignments/202604/$dataset"
  remote="s3://bec.bdrc.io/ocr_benchmark/alignments/202604/$dataset"
  for parquet in "$local"/*.parquet; do
    aws s3 cp "$parquet" "$remote/$(basename "$parquet")" --only-show-errors
  done
  aws s3 cp "$local/catalog_alignments.csv" "$remote/catalog_alignments.csv" --only-show-errors
  aws s3 cp "$local/catalog_volumes.csv" "$remote/catalog_volumes.csv" --only-show-errors
  aws s3 cp "$local/README.md" "$remote/README.md" --only-show-errors
done
aws s3 sync "$OUT/reports/" "${RUN_URI}reports/" --only-show-errors
aws s3 cp "$OUT/render_manifest.parquet" "${RUN_URI}render_manifest.parquet" --only-show-errors
aws s3 cp "$OUT/render_plan.parquet" "${RUN_URI}render_plan.parquet" --only-show-errors
aws s3 cp "$OUT/document_augmentation_manifest.json" \
  "${RUN_URI}document_augmentation_manifest.json" --only-show-errors

echo "[catalog] Updating shared dataset totals to 13 volumes / 13,000 pages"
"$PYTHON" - <<'PY'
import csv
import io
import subprocess

uri = "s3://bec.bdrc.io/ocr_benchmark/alignments/202604/datasets.csv"
text = subprocess.run(
    ["aws", "s3", "cp", uri, "-"], check=True, capture_output=True, text=True
).stdout
reader = csv.DictReader(io.StringIO(text))
fieldnames = list(reader.fieldnames or [])
rows = list(reader)
targets = {"BECSynthetic_RetainEN", "BECSynthetic_RetainZH"}
seen = set()
for row in rows:
    if row.get("dataset_id") in targets:
        row["nb_img_volumes"] = "13"
        row["nb_img_files"] = "13000"
        seen.add(row["dataset_id"])
if seen != targets:
    raise SystemExit(f"Missing shared dataset rows: {sorted(targets - seen)}")
with open("/tmp/datasets.retention-batch2.csv", "w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY
aws s3 cp /tmp/datasets.retention-batch2.csv \
  s3://bec.bdrc.io/ocr_benchmark/alignments/202604/datasets.csv --only-show-errors

printf 'instance_id=%s\ncompleted_at=%s\npages=16000\nvolumes=0006-0013\n' \
  "$(hostname)" "$(date --iso-8601=seconds)" |
  aws s3 cp - "${RUN_URI}_SUCCESS" --only-show-errors

echo "[done] Retention Batch 2 uploaded; stopping instance"
