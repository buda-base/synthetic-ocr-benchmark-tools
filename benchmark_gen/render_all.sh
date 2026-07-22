#!/usr/bin/env bash
set -uo pipefail

FONTS_CSV="${1:-digital_fonts.filtered.csv}"   # allow override
OUTDIR="out"
LOGDIR="$OUTDIR/latex_logs"
SUMMARY="$OUTDIR/render_summary.csv"

mkdir -p "$OUTDIR" "$LOGDIR"

echo "basename,status,latex_exit,notes" > "$SUMMARY"

render_one () {
  local basename="$1"
  local dpi="$2"

  local tex_path="$OUTDIR/${basename}.tex"
  local pdf_path="$OUTDIR/${basename}.pdf"
  local jpg_path="$OUTDIR/${basename}.jpg"
  local log_path="$LOGDIR/${basename}.log"
  local err_path="$LOGDIR/${basename}.err"
  local pdftoppm_err_path="$LOGDIR/${basename}.pdftoppm.err"

  if [[ ! -f "$tex_path" ]]; then
    echo "!! Missing TeX for $basename ($tex_path). Skipping."
    echo "$basename,skip,NA,missing_tex" >> "$SUMMARY"
    return 0
  fi

  local latex_exit=0
  if [[ -f "$pdf_path" ]]; then
    echo "PDF already exists for $basename, skipping LaTeX"
  else
    echo "LaTeX $basename"

    lualatex \
      -interaction=nonstopmode \
      -halt-on-error \
      -file-line-error \
      -output-directory="$OUTDIR" \
      "$tex_path" \
      >"$log_path" 2>"$err_path"
    latex_exit=$?

    if [[ $latex_exit -ne 0 || ! -f "$pdf_path" ]]; then
      echo "!! LaTeX FAILED for $basename (exit=$latex_exit)"
      echo "   log: $log_path"
      echo "   --- tail log ---"
      tail -n 20 "$log_path" | sed 's/^/   /'
      echo "   ---------------"
      echo "$basename,fail,$latex_exit,latex_error" >> "$SUMMARY"
      return 0
    fi
  fi

  echo "JPG  $basename"

  # Skip JPG generation if it already exists
  if [[ -f "$jpg_path" ]]; then
    echo "JPG already exists for $basename, skipping pdftoppm"
    echo "$basename,ok,$latex_exit," >> "$SUMMARY"
    return 0
  fi

  # Check PDF exists and is readable before pdftoppm
  if [[ ! -f "$pdf_path" ]]; then
    echo "========================================"
    echo "!! ERROR: PDF missing for $basename"
    echo "   Expected: $pdf_path"
    echo "========================================"
    echo "$basename,fail,$latex_exit,missing_pdf" >> "$SUMMARY"
    return 0
  fi

  local pdf_size=$(stat -f%z "$pdf_path" 2>/dev/null || stat -c%s "$pdf_path" 2>/dev/null || echo "unknown")
  if [[ "$pdf_size" == "0" ]]; then
    echo "========================================"
    echo "!! ERROR: PDF is empty (0 bytes) for $basename"
    echo "   Path: $pdf_path"
    echo "========================================"
    echo "$basename,fail,$latex_exit,empty_pdf" >> "$SUMMARY"
    return 0
  fi

  # Run pdftoppm and capture stderr
  if ! pdftoppm -f 1 -l 1 -jpeg -jpegopt quality=85 -gray -scale-to-x 1200 -scale-to-y -1 \
        "$pdf_path" "$OUTDIR/${basename}" \
        > /dev/null 2>"$pdftoppm_err_path"; then
    echo "========================================"
    echo "!! ERROR: pdftoppm FAILED for $basename"
    echo "   PDF: $pdf_path (size: $pdf_size bytes)"
    echo "   Error log: $pdftoppm_err_path"
    echo "   --- pdftoppm error output ---"
    cat "$pdftoppm_err_path" | sed 's/^/   /'
    echo "   -----------------------------"
    echo "========================================"
    echo "$basename,fail,$latex_exit,pdftoppm_error" >> "$SUMMARY"
    return 0
  fi

  # Check if expected JPG was created (pdftoppm may use -1.jpg or -01.jpg depending on page count)
  local jpg_found=""
  if [[ -f "$OUTDIR/${basename}-1.jpg" ]]; then
    jpg_found="$OUTDIR/${basename}-1.jpg"
  elif [[ -f "$OUTDIR/${basename}-01.jpg" ]]; then
    jpg_found="$OUTDIR/${basename}-01.jpg"
  fi

  if [[ -n "$jpg_found" ]]; then
    mv "$jpg_found" "$jpg_path"
  else
    echo "========================================"
    echo "!! ERROR: pdftoppm produced no JPG for $basename"
    echo "   PDF: $pdf_path (size: $pdf_size bytes)"
    echo "   Expected output: $OUTDIR/${basename}-1.jpg or $OUTDIR/${basename}-01.jpg"
    echo "   pdftoppm stderr: $pdftoppm_err_path"
    if [[ -f "$pdftoppm_err_path" ]] && [[ -s "$pdftoppm_err_path" ]]; then
      echo "   --- pdftoppm error output ---"
      cat "$pdftoppm_err_path" | sed 's/^/   /'
      echo "   -----------------------------"
    else
      echo "   (no error output from pdftoppm)"
    fi
    echo "   --- files in output dir matching pattern ---"
    ls -lh "$OUTDIR/${basename}"* 2>/dev/null | sed 's/^/   /' || echo "   (no matching files found)"
    echo "========================================"
    echo "$basename,fail,$latex_exit,no_jpg" >> "$SUMMARY"
    return 0
  fi

  echo "$basename,ok,$latex_exit," >> "$SUMMARY"
  return 0
}

# Correct CSV parsing: feed CSV as stdin *to python code passed via -c*
python3 -c '
import csv, sys
r = csv.DictReader(sys.stdin)
for row in r:
    b = row.get("basename","").strip()
    dpi = row.get("dpi","300").strip()
    if b:
        print(b + "\t" + dpi)
' < "$FONTS_CSV" | while IFS=$'\t' read -r basename dpi; do
  render_one "$basename" "$dpi"
done

echo
echo "Done. Summary: $SUMMARY"
echo "Logs: $LOGDIR/"
