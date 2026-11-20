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
OUT=/mnt/synthbench/retention_en_zh_batch1
CACHE=/mnt/synthbench/retention_cache
LOG=/mnt/synthbench/retention-en-zh-batch1.log
PAYLOAD_URI=s3://bec.bdrc.io/synthetic/staging/retention_en_zh_batch1_payload.tar.zst
RUN_URI=s3://bec.bdrc.io/synthetic/benchmarks/retention_en_zh_batch1/
EVAL_URI=s3://bec.bdrc.io/evaluation_benchmark/eval_parquets/yigdzin2_dev/dev-retain.parquet
EVAL=/mnt/synthbench/dev-retain.parquet

mkdir -p /mnt/synthbench "$PROJECT" "$CACHE"
exec > >(tee -a "$LOG") 2>&1

finish() {
  status=$?
  set +e
  aws s3 cp "$LOG" "${RUN_URI}retention-en-zh-batch1.log" --only-show-errors
  if [[ $status -ne 0 ]]; then
    printf 'instance_id=%s\nfailed_at=%s\nexit_code=%s\n' \
      "$(hostname)" "$(date --iso-8601=seconds)" "$status" |
      aws s3 cp - "${RUN_URI}_FAILED" --only-show-errors
  fi
  shutdown -h now
  exit "$status"
}
trap finish EXIT

aws s3 rm "${RUN_URI}_FAILED" --only-show-errors || true

echo "[bootstrap] Installing renderer and licensed font packages"
apt-get update
apt-get install -y --no-install-recommends \
  awscli fontconfig fonts-arphic-ukai fonts-arphic-uming fonts-dejavu-core \
  fonts-ebgaramond fonts-liberation2 fonts-noto-cjk \
  fonts-noto-cjk-extra fonts-noto-core fonts-wqy-microhei fonts-wqy-zenhei \
  libgl1 libglib2.0-0 libharfbuzz0b libfribidi0 python3-pip python3-venv zstd
fc-cache -f

aws s3 cp "$PAYLOAD_URI" /tmp/retention_en_zh_batch1_payload.tar.zst
tar --zstd -xf /tmp/retention_en_zh_batch1_payload.tar.zst -C "$PROJECT"

if [[ ! -x "$PYTHON" ]]; then
  python3 -m venv /opt/synthbench-venv
fi
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r "$PROJECT/synthetic_benchmark/requirements-render.txt"

echo "[inputs] Downloading evaluation text only for near-duplicate rejection"
aws s3 cp "$EVAL_URI" "$EVAL" --only-show-errors

echo "[smoke] Rendering two pages per language through the complete pipeline"
rm -rf /mnt/synthbench/retention_smoke
cd "$PROJECT"
PYTHONPATH="$PROJECT/synthetic_benchmark" "$PYTHON" \
  synthetic_benchmark/render_retention_en_zh.py \
  --out-dir /mnt/synthbench/retention_smoke \
  --eval-parquet "$EVAL" \
  --count-per-language 2 \
  --source-docs-per-variant 40 \
  --cache-dir "$CACHE" \
  --jobs 2
rm -rf /mnt/synthbench/retention_smoke

echo "[render] Rendering 5,000 English + 5,000 Chinese pages"
rm -rf "$OUT"
PYTHONPATH="$PROJECT/synthetic_benchmark" "$PYTHON" \
  synthetic_benchmark/render_retention_en_zh.py \
  --out-dir "$OUT" \
  --eval-parquet "$EVAL" \
  --count-per-language 5000 \
  --source-docs-per-variant 300 \
  --cache-dir "$CACHE" \
  --jobs 12

echo "[validate] Checking image/alignment cardinality and target conventions"
"$PYTHON" - "$OUT" <<'PY'
import sys
from pathlib import Path
import pyarrow.parquet as pq

root = Path(sys.argv[1])
expected = {
    "BECSynthetic_RetainEN": 5000,
    "BECSynthetic_RetainZH": 5000,
}
for dataset, count in expected.items():
    parquet_files = sorted((root / "alignments" / "202604" / dataset).glob("*.parquet"))
    rows = sum(pq.read_table(path).num_rows for path in parquet_files)
    if rows != count:
        raise SystemExit(f"{dataset}: expected {count} rows, found {rows}")
    for path in parquet_files:
        for row in pq.read_table(path, columns=["img_file_name", "transcription"]).to_pylist():
            text = row["transcription"]
            if text != text.rstrip() or any(line.endswith(" ") for line in text.splitlines()):
                raise SystemExit(f"Invalid target whitespace in {path}: {row['img_file_name']}")
images = list((root / "images").glob("**/*.jpg")) + list((root / "images").glob("**/*.tif"))
if len(images) != 10000:
    raise SystemExit(f"Expected 10,000 images, found {len(images)}")
print("Validated 10,000 images and alignment rows")
PY

echo "[upload] Images, alignments, and report"
for w_id in W1BCS007 W1BCS008; do
  aws s3 sync "$OUT/images/$w_id/" \
    "s3://bec.bdrc.io/ocr_benchmark/images/$w_id/" --only-show-errors
done
for dataset in BECSynthetic_RetainEN BECSynthetic_RetainZH; do
  aws s3 sync "$OUT/alignments/202604/$dataset/" \
    "s3://bec.bdrc.io/ocr_benchmark/alignments/202604/$dataset/" --only-show-errors
done
aws s3 sync "$OUT/reports/" "${RUN_URI}reports/" --only-show-errors
aws s3 cp "$OUT/render_manifest.parquet" "${RUN_URI}render_manifest.parquet" --only-show-errors
aws s3 cp "$OUT/render_plan.parquet" "${RUN_URI}render_plan.parquet" --only-show-errors
aws s3 cp "$OUT/document_augmentation_manifest.json" \
  "${RUN_URI}document_augmentation_manifest.json" --only-show-errors

echo "[catalog] Upserting both datasets in shared datasets.csv"
"$PYTHON" - "$OUT/alignments/202604/datasets.csv" <<'PY'
import csv
import io
import subprocess
import sys

shared_uri = "s3://bec.bdrc.io/ocr_benchmark/alignments/202604/datasets.csv"
local_path = sys.argv[1]
existing = subprocess.run(
    ["aws", "s3", "cp", shared_uri, "-"], check=True, capture_output=True, text=True
).stdout
reader = csv.DictReader(io.StringIO(existing))
fieldnames = list(reader.fieldnames or [])
rows = list(reader)
with open(local_path, encoding="utf-8", newline="") as stream:
    additions = list(csv.DictReader(stream))
for addition in additions:
    replacement = {name: addition.get(name, "") for name in fieldnames}
    for index, row in enumerate(rows):
        if row.get("dataset_id") == addition["dataset_id"]:
            rows[index] = replacement
            break
    else:
        rows.append(replacement)
buf = io.StringIO()
writer = csv.DictWriter(buf, fieldnames=fieldnames)
writer.writeheader()
writer.writerows(rows)
with open("/tmp/datasets.retention-merged.csv", "w", encoding="utf-8") as stream:
    stream.write(buf.getvalue())
PY
aws s3 cp /tmp/datasets.retention-merged.csv \
  s3://bec.bdrc.io/ocr_benchmark/alignments/202604/datasets.csv --only-show-errors

printf 'instance_id=%s\ncompleted_at=%s\npages=10000\n' \
  "$(hostname)" "$(date --iso-8601=seconds)" |
  aws s3 cp - "${RUN_URI}_SUCCESS" --only-show-errors

echo "[done] Retention batch uploaded; stopping instance"
