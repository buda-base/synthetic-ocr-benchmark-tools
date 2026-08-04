#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
# Augraphy's Folding helper currently cannot JIT-compile under recent Python.
# Its NumPy implementation is fast enough and avoids the Numba dispatcher crash.
export NUMBA_DISABLE_JIT=1
# Batches render in separate processes (see render_batches_parallel). Pin each
# process to a single math thread so --jobs maps to cores instead of
# oversubscribing them via nested BLAS/OpenMP/OpenCV thread pools.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_NUM_THREADS=1

PROJECT=/mnt/synthbench/repo
PYTHON=/opt/synthbench-venv/bin/python
export PYTHONPATH="$PROJECT/synthetic_benchmark:$PROJECT/coverage_report"

PLAN="$PROJECT/full_benchmark_plan_40000_v5.parquet"
SUPPORT="$PROJECT/coverage_report/out/stack_support_v3.parquet"
CHUNKS="$PROJECT/synthetic_benchmark/out/bocorpus_chunks.parquet"
OUT=/mnt/synthbench/full_benchmark_40000_v5
LOG=/mnt/synthbench/full-benchmark-40000-v5.log

MW_ID=MW1BCS005
W_ID=W1BCS005
IE_ID=IE1BCS005
I_ID_PREFIX=I1BCS005
VE_ID_PREFIX=VE1BCS005
# Benchmark/release version shared across all datasets in the release (the same
# 202604 release that holds BECSynthetic_01..04 and the non-synthetic sets).
# This is NOT bumped per synthetic run; only DATASET_ID advances.
VOLUME_VERSION=v001
BENCHMARK_VERSION=202604
DATASET_ID=BECSynthetic_05

IMAGE_URI="s3://bec.bdrc.io/ocr_benchmark/images/$W_ID/"
ALIGNMENT_URI="s3://bec.bdrc.io/ocr_benchmark/alignments/$BENCHMARK_VERSION/$DATASET_ID/"
SHARED_DATASETS_URI="s3://bec.bdrc.io/ocr_benchmark/alignments/$BENCHMARK_VERSION/datasets.csv"
RUN_URI=s3://bec.bdrc.io/synthetic/benchmarks/full_40000_v5/

exec > >(tee -a "$LOG") 2>&1
cd "$PROJECT"
mkdir -p "$OUT"

if [[ -s "$PLAN" ]]; then
  echo "[plan] Reusing existing v5 plan: $PLAN"
else
  echo "[plan] Building 40,000-page v5 plan: 4-way script balance (10,000 each), 10% short/long, 7% repetitive + 3% highly repetitive"
  mkdir -p "$(dirname "$CHUNKS")"
  "$PYTHON" synthetic_benchmark/build_render_plan.py \
    --output "$PLAN" \
    --chunks "$CHUNKS" \
    --support-parquet "$SUPPORT" \
    --target-images 40000 \
    --balance-by script_4 \
    --top-up-category-quotas \
    --max-text-repetition-score 0.50 \
    --min-zstd-compression-ratio 0.18 \
    --repetitive-fraction 0.07 \
    --highly-repetitive-fraction 0.03 \
    --repetitive-min-score 0.30 \
    --highly-repetitive-min-score 0.50 \
    --short-length-fraction 0.10 \
    --long-length-fraction 0.10 \
    --seed 13
fi

echo "[plan] Verifying v5 plan: $PLAN"
"$PYTHON" - "$PLAN" <<'PY'
import collections
import sys

import pyarrow.parquet as pq

from shorthand_aug import syllable_count

rows = pq.read_table(
    sys.argv[1],
    columns=[
        "script_4",
        "text_length_bucket",
        "text",
        "source_text_repetition_policy",
        "source_text_repetition_score",
    ],
).to_pylist()
if len(rows) != 40_000:
    raise SystemExit(f"Expected 40,000 plan rows, found {len(rows)}")
cats = collections.Counter(str(row["script_4"]) for row in rows)
expected = {
    "uchen": 10_000,
    "druma": 10_000,
    "danyig_pedri": 10_000,
    "gyuyig_tsugdri": 10_000,
}
if dict(cats) != expected:
    raise SystemExit(f"Unexpected script_4 quotas: {dict(sorted(cats.items()))}")
buckets = collections.Counter(
    str(row.get("text_length_bucket") or "normal") for row in rows
)
rep_policies = collections.Counter(
    str(row.get("source_text_repetition_policy") or "penalized") for row in rows
)
shorts = [row for row in rows if row.get("text_length_bucket") == "short"]
longs = [row for row in rows if row.get("text_length_bucket") == "long"]
if buckets.get("short", 0) < 3500 or buckets.get("long", 0) < 3500:
    raise SystemExit(f"Length buckets under-filled: {dict(buckets)}")
if rep_policies.get("repetitive", 0) < 2500:
    raise SystemExit(f"Repetitive slots under-filled: {dict(rep_policies)}")
if rep_policies.get("highly_repetitive", 0) < 1000:
    raise SystemExit(f"Highly-repetitive slots under-filled: {dict(rep_policies)}")
mid = [
    row
    for row in rows
    if row.get("source_text_repetition_policy") == "repetitive"
]
high = [
    row
    for row in rows
    if row.get("source_text_repetition_policy") == "highly_repetitive"
]
if any(float(row["source_text_repetition_score"]) < 0.30 for row in mid):
    raise SystemExit("Repetitive slots include scores below 0.30")
if any(float(row["source_text_repetition_score"]) < 0.50 for row in high):
    raise SystemExit("Highly-repetitive slots include scores below 0.50")
bad_short = [
    row
    for row in shorts
    if (not str(row["text"]).rstrip().endswith("།"))
    or syllable_count(str(row["text"])) > 10
]
if bad_short:
    raise SystemExit(f"{len(bad_short)} short rows fail shad/syllable constraints")
bad_long = [row for row in longs if syllable_count(str(row["text"])) < 1000]
if bad_long:
    raise SystemExit(f"{len(bad_long)} long rows have <1000 syllables")
print(f"[plan] Verified 40,000 rows; script_4={dict(sorted(cats.items()))}")
print(f"[plan] Length buckets: {dict(sorted(buckets.items()))}")
print(f"[plan] Repetition policies: {dict(sorted(rep_policies.items()))}")
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

echo "[render] Starting 40,000-page v5 production benchmark (force rebuild)"
# Drop prior physical outputs so every image is re-rendered under the current
# augmentation and resolution policy.
rm -rf "$OUT"
mkdir -p "$OUT"
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
  --jobs 12 \
  --force

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
try:
    existing_text = subprocess.run(
        ["aws", "s3", "cp", shared_uri, "-"],
        check=True, capture_output=True, text=True,
    ).stdout
except subprocess.CalledProcessError:
    existing_text = ""
if existing_text.strip():
    existing_reader = csv.DictReader(io.StringIO(existing_text))
    fieldnames = list(existing_reader.fieldnames or [])
    rows = list(existing_reader)
else:
    # Shared catalog does not yet exist for this benchmark version; seed it from
    # the local dataset header.
    with open(local_path, encoding="utf-8", newline="") as stream:
        fieldnames = list(csv.DictReader(stream).fieldnames or [])
    rows = []
if "dataset_id" not in fieldnames:
    raise SystemExit(f"Unexpected shared catalog header: {fieldnames}")
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
echo "[done] v5 full benchmark complete: $RUN_URI"
