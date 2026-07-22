# Synthetic BoCorpus Benchmark

Pipeline for generating a large synthetic Tibetan OCR benchmark from OpenPecha BoCorpus text and the font catalog used by `scripts/benchmark_gen/`.

The output format is:

```text
out/dataset/
  images/W1BCS001/I1BCS001_0001/v001/0001.jpg
  alignments/202604/datasets.csv
  alignments/202604/BECSynthetic_01/
    I1BCS001_0001-VE1BCS001_0001_ptt.parquet
    catalog_alignments.csv
    catalog_volumes.csv
    README.md
  checkpoints/catalog_batches/
```

Images are rendered as grayscale JPEGs with quality 80 and width 2400 px. The page shape is pecha-like by default: width is four times height.

## Inputs

- BoCorpus parquet: defaults to `scripts/coverage_report/.cache/bocorpus/bo_corpus.parquet` and is downloaded from Hugging Face if missing.
- Font metadata: `scripts/benchmark_gen/catalog/Benchmark catalog - digital_fonts.csv`, `scripts/benchmark_gen/catalog/Script lists - Scripts.csv`, and `scripts/benchmark_gen/digital_fonts.filtered.csv`.
- Stack support: pass the rebuilt coverage parquet from `scripts/coverage_report/build_support_dataset.py`.

The font filter excludes only `script_id=239`, which is the decorative digital-font subtype inside `Parma (Printed Scripts)`. The render plan balances the target image count across the 7 distinct values currently present in the catalog's `8 categories` column.

## 1. Build BoCorpus Chunks

```bash
/home/eroux/pvenvs/1/bin/python synthetic_benchmark/build_bocorpus_chunks.py \
  --output synthetic_benchmark/out/bocorpus_chunks.parquet
```

Useful smoke-test options:

```bash
/home/eroux/pvenvs/1/bin/python synthetic_benchmark/build_bocorpus_chunks.py \
  --limit-rows 200 \
  --limit-chunks 1000
```

Chunks are page-sized text samples with `bocorpus_row`, `char_start`, `char_end`, text, and stack sets. The default chunking targets roughly 1300 characters, with 900 and 1800 character soft bounds. Text is normalized with the same Botok Unicode and graphical normalization used by the coverage scripts. Chunks containing five consecutive tshegs (`U+0F0B`) are skipped because they are usually table-of-contents leader lines.

Pass `--force-download` to refresh the cached BoCorpus parquet, or `--bocorpus-parquet /path/to/bo_corpus.parquet` to use a local file.

## 1b. Shorthand lexicon + coverage probes (optional linguistic augmentation)

```bash
/home/eroux/pvenvs/1/bin/python synthetic_benchmark/build_shorthand_lexicon.py
/home/eroux/pvenvs/1/bin/python coverage_report/export_shorthand_stacks.py
/home/eroux/pvenvs/1/bin/python coverage_report/build_support_dataset.py --mode both
```

Then review auto-ok shorthand renderings before enabling injection:

```bash
/home/eroux/pvenvs/1/bin/python coverage_report/render_shorthand_audit.py \
  coverage_report/out/stack_support.parquet \
  --kind shorthand-pass \
  --sample-size 80
```

Reject bad cases via heuristics or `data/shorthands/denylist.csv`. Shorthand injection stays off until you pass `--enable-shorthands` at render time.

## 2. Build Render Plan

```bash
/home/eroux/pvenvs/1/bin/python synthetic_benchmark/build_render_plan.py \
  --chunks synthetic_benchmark/out/bocorpus_chunks.parquet \
  --support-parquet coverage_report/out/stack_support.parquet \
  --target-images 500000 \
  --output synthetic_benchmark/out/render_plan.parquet
```

For dense ume shorthand pages (every other ume image), also pass `--oversample-ume-dense` so those plan rows start from ~2× source text before contraction.

The planner rejects a `(font, chunk)` candidate if any Tibetan stack in the whole chunk is unsupported by that font. Unknown stacks are treated as unsupported. It uses `ok=True` and, when present, `placement_warning_count=0`.

## 3. Render Pecha JPEG/Alignment Pairs

```bash
/home/eroux/pvenvs/1/bin/python synthetic_benchmark/render_batches.py \
  synthetic_benchmark/out/render_plan.parquet \
  --out-dir synthetic_benchmark/out/dataset \
  --batch-size 100 \
  --jobs 4
```

After the shorthand coverage review gate, enable linguistic augmentation:

```bash
/home/eroux/pvenvs/1/bin/python synthetic_benchmark/render_batches.py \
  synthetic_benchmark/out/render_plan.parquet \
  --enable-shorthands \
  --support-parquet coverage_report/out/stack_support.parquet \
  --out-dir synthetic_benchmark/out/dataset
```

Script policy:

- **uchen**: sparse random replacements, capped at about 4 shorthands / 100 syllables (often fewer / sometimes none)
- **ume**: dense replacements on even `image_id` pages; none on odd pages

Only shorthands whose stacks are supported by the paired font (and not denylisted) are applied. Ground-truth transcriptions use the contracted text.

The renderer groups pages by font into multi-page LuaLaTeX batches, then rasterizes with:

```text
pdftoppm -jpeg -jpegopt quality=80 -gray -scale-to-x 2400 -scale-to-y -1
```

The final JPEGs are explicitly converted to grayscale (`L`) before being written.

Pecha page defaults:

```text
page_height = 74 mm
page_width = 4 * page_height = 296 mm
left/right margins = 20 mm
top/bottom margins = 16 mm
font scale = 1.5 * font_size_pt
page prefix = ༄༅། ། on every other exported page, starting with page 1
```

Override with `--page-height-mm`, `--page-ratio`, `--margin-x-mm`, `--margin-y-mm`, `--font-scale`, `--image-width-px`, `--page-prefix`, and `--no-page-prefix`.

Use `--jobs N` to render batches in parallel. Each worker renders into an isolated temporary directory under `out/dataset/workers/`; the parent process moves completed batch outputs into the local benchmark tree and writes small per-batch checkpoint fragments under `out/dataset/checkpoints/catalog_batches/`.

Rendering is resumable by default. On startup, the renderer reads the per-batch checkpoint fragments, skips plan rows whose image files already exist, then continues at the next output sequence. Use `--force` to ignore existing checkpoints and regenerate the requested rows. If a run is interrupted, progress is preserved up to the last checkpointed completed batch. The final alignment parquet files and benchmark catalogs are rebuilt at normal completion.

During LaTeX shipout, the generated TeX writes `batches/<batch>.pages.csv` with:

```text
physical_page, render_id, line_count
```

It also writes `batches/<batch>.lines.csv` with marker IDs per physical rendered line. The renderer uses those IDs to rebuild each transcription with line breaks matching the exported image, using the same marker approach as `scripts/benchmark_gen`.

This lets the Python renderer know how many physical PDF pages each chunk used. Each render-plan row starts as one output image. If a rendered chunk flows to more than one physical page, only its first page is exported as a benchmark image and transcription; later overflow pages are ignored, and the next chunk is still found from the page map. If a chunk uses one page with `--min-lines-per-image` lines or fewer, default `5`, it is merged with the next chunk and the batch is re-rendered. Per-batch checkpoint fragments record render diagnostics such as `physical_pages_for_chunk`, `first_page_line_count`, rendered font size, and pipe-separated source chunk IDs when chunks were merged.

Every other exported page gets `༄༅། །` prepended before TeX rendering and transcription, unless `༄` already appears in the first five characters. This starts with output page `1`; use `--no-page-prefix` to disable it.

## 4. Validate

```bash
/home/eroux/pvenvs/1/bin/python synthetic_benchmark/validate_output.py \
  synthetic_benchmark/out/dataset
```

This checks that every alignment row has its image file, samples image dimensions and image modes, and reports alignment parquet counts.

## Notes

- Alignment parquet files contain the rendered transcription for each exported first physical page, with line breaks reconstructed from TeX line markers. If a chunk overflows to later physical pages, text on ignored overflow pages is not included.
- Rendering failures are batch-level; failed batches are logged under `out/dataset/logs/`.
- For a full 500k run, start with `--limit` on `render_batches.py` to test the local TeX/Poppler setup before running all batches.

