#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
# Augraphy's Folding helper currently cannot JIT-compile under recent Python.
# Its NumPy implementation is fast enough and avoids the Numba dispatcher crash.
export NUMBA_DISABLE_JIT=1

PROJECT=/mnt/synthbench/repo
PYTHON=/opt/synthbench-venv/bin/python
export PYTHONPATH="$PROJECT/synthetic_benchmark:$PROJECT/coverage_report"

PLAN="$PROJECT/full_benchmark_plan_45714_v3_relaxed.parquet"
SUPPORT="$PROJECT/coverage_report/out/stack_support_v3.parquet"
OUT=/mnt/synthbench/full_benchmark_45714_v3
LOG=/mnt/synthbench/full-benchmark-45714-v3.log

MW_ID=MW1BCS003
W_ID=W1BCS003
IE_ID=IE1BCS003
I_ID_PREFIX=I1BCS003
VE_ID_PREFIX=VE1BCS003
VOLUME_VERSION=v001
BENCHMARK_VERSION=202604
DATASET_ID=BECSynthetic_03

IMAGE_URI="s3://bec.bdrc.io/ocr_benchmark/images/$W_ID/"
ALIGNMENT_URI="s3://bec.bdrc.io/ocr_benchmark/alignments/$BENCHMARK_VERSION/$DATASET_ID/"
SHARED_DATASETS_URI="s3://bec.bdrc.io/ocr_benchmark/alignments/$BENCHMARK_VERSION/datasets.csv"
RUN_URI=s3://bec.bdrc.io/synthetic/benchmarks/full_45714_v3/

exec > >(tee -a "$LOG") 2>&1
cd "$PROJECT"
mkdir -p "$OUT"

echo "[plan] Verifying v3 relaxed plan: $PLAN"
"$PYTHON" - "$PLAN" <<'PY'
import collections
import sys

import pyarrow.parquet as pq

rows = pq.read_table(sys.argv[1], columns=["script_8"]).to_pylist()
if len(rows) != 45_714:
    raise SystemExit(f"Expected 45,714 plan rows, found {len(rows)}")
counts = collections.Counter(str(row["script_8"]) for row in rows)
print(f"[plan] Verified 45,714 rows; script_8 quotas: {dict(sorted(counts.items()))}")
PY

periodic_image_upload() {
  while true; do
    if [[ -d "$OUT/images/$W_ID" ]]; then
      echo "[upload] Periodic image synchronization"
      aws s3 sync "$OUT/images/$W_ID/" "$IMAGE_URI" --only-show-errors || true
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

echo "[render] Starting 45,714-page v3 production benchmark"
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
  --support-parquet "$SUPPORT" \
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

echo "[catalog] Safely upserting $DATASET_ID into shared $BENCHMARK_VERSION/datasets.csv"
"$PYTHON" - "$SHARED_DATASETS_URI" "$OUT/alignments/$BENCHMARK_VERSION/datasets.csv" "$DATASET_ID" <<'PY'
import csv
import io
import subprocess
import sys

shared_uri, local_path, dataset_id = sys.argv[1], sys.argv[2], sys.argv[3]

# Existing shared catalog (7 columns incl. line_breaks); preserve every other row.
existing_text = subprocess.run(
    ["aws", "s3", "cp", shared_uri, "-"],
    check=True, capture_output=True, text=True,
).stdout
existing_reader = csv.DictReader(io.StringIO(existing_text))
fieldnames = list(existing_reader.fieldnames or [])
if "dataset_id" not in fieldnames:
    raise SystemExit(f"Unexpected shared catalog header: {fieldnames}")
rows = list(existing_reader)

# Locally generated single-row datasets.csv for this run (6 columns, no line_breaks).
with open(local_path, encoding="utf-8", newline="") as stream:
    local_row = next(csv.DictReader(stream))

new_row = {name: "" for name in fieldnames}
for key in ("dataset_id", "nb_img_volumes", "nb_img_files", "technology", "script", "modes"):
    if key in new_row:
        new_row[key] = local_row.get(key, "")
if "line_breaks" in new_row:
    new_row["line_breaks"] = "Y"

replaced = False
for i, row in enumerate(rows):
    if row.get("dataset_id") == dataset_id:
        rows[i] = new_row
        replaced = True
        break
if not replaced:
    rows.append(new_row)

buf = io.StringIO()
writer = csv.DictWriter(buf, fieldnames=fieldnames)
writer.writeheader()
writer.writerows(rows)
merged = buf.getvalue()

# Keep a local backup of both the pre-merge shared catalog and the merged result.
with open("/tmp/datasets.pre_merge.csv", "w", encoding="utf-8") as f:
    f.write(existing_text)
with open("/tmp/datasets.merged.csv", "w", encoding="utf-8") as f:
    f.write(merged)

action = "Replaced" if replaced else "Appended"
print(f"[catalog] {action} {dataset_id}: {new_row}")
print(f"[catalog] Shared catalog now lists {len(rows)} datasets")
PY

aws s3 cp /tmp/datasets.merged.csv "$SHARED_DATASETS_URI" --only-show-errors
echo "[catalog] Uploaded merged shared datasets.csv"

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
printf 'instance_id=%s\ncompleted_at=%s\nplanned_rows=45714\nphysical_images=%s\nw_id=%s\nie_id=%s\ndataset_id=%s\n' \
  "$(hostname)" \
  "$(date --iso-8601=seconds)" \
  "$PHYSICAL_IMAGES" \
  "$W_ID" \
  "$IE_ID" \
  "$DATASET_ID" |
  aws s3 cp - "${RUN_URI}_SUCCESS" --only-show-errors

cleanup_uploader
trap - EXIT
echo "[done] v3 full benchmark complete: $RUN_URI"
