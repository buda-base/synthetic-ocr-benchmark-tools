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

Image widths are deterministically randomized from 1800 to 3500 px. JPEG quality
is 85 by default, with 10% of JPEG pages at quality 65. Clean pages use grayscale
JPEG; augmented pages can also use RGB JPEG or bilevel Group 4 TIFF according to
their paper background. The page shape is pecha-like by default: width is four
times height.

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

Small with/without smoke plan (~2 images per font; even=`with`, odd=`none`):

```bash
/home/eroux/pvenvs/1/bin/python synthetic_benchmark/build_render_plan.py \
  --chunks synthetic_benchmark/out/bocorpus_chunks.parquet \
  --support-parquet coverage_report/out/stack_support.parquet \
  --images-per-font 2 \
  --pair-shorthand-modes \
  --oversample-ume-dense \
  --output synthetic_benchmark/out/render_plan_shorthand_smoke.parquet
```

The planner rejects a `(font, chunk)` candidate if any Tibetan stack in the
whole chunk is unsupported by that font. Unknown stacks are treated as
unsupported. It uses `ok=True` and, when present,
`placement_warning_count=0`. Within every font face, it alternates source-text
tiers so that half of the selected chunks contain no rare stacks (`normal`) and
half prioritize stacks above the corpus rarity threshold (`difficult`). If a
font cannot render enough rare-stack chunks, the difficult half uses its most
structurally complex coverage-compatible common chunks instead. The rarity tier
and fallback basis are recorded separately, before shorthand or other text
augmentation.

Repetition is balanced independently from rarity. For each font, nine out of
every ten source-text slots prefer the least repetitive compatible chunks; the
tenth deliberately prefers highly repetitive text so that 10% remains covered.
The score compares dominant syllables, adjacent repeats, and repeated bigrams
and trigrams. The score and requested policy are stored in the render plan.

## 2b. Review font-level augmentation

Font augmentation is experimental and remains separate from dataset rendering until
the generated forms pass manual Tibetan review. Generate the first conservative,
single-axis review batch with:

```bash
/home/eroux/pvenvs/1/bin/python synthetic_benchmark/render_font_augmentation_audit.py
```

This selects a taxonomy-stratified sample of fonts and writes a contact sheet,
JSON manifest, editable `review.csv`, individual PNGs, and cached font variants
under `synthetic_benchmark/out/font_augmentation_review/round1/`.

Each contact-sheet row shows the unmodified face followed by mild width, slant,
and stroke-weight variants. Automated HarfBuzz checks reject any variant that
introduces a shaping or placement regression on a source-supported probe. Record
`accept`, `borderline`, or `reject` in `review.csv`, with optional reason tags
and notes. Re-running the command preserves those three review columns.

Use `--font BASENAME` (repeatable) to review specific faces, or `--max-fonts N`
to change the default batch size. The reviewed probe set is
`data/font_augmentation/probes.txt`. Variants are deterministic and cached by
source-font hash, TTC face index, operation, and parameter.

After reviewing round 1, generate the wider-width, stronger negative-slant grid
with:

```bash
/home/eroux/pvenvs/1/bin/python synthetic_benchmark/render_font_augmentation_audit.py \
  --preset round2
```

Round 2 tests width scales `0.80`, `0.88`, `1.15`, and `1.25`, plus slants of
`-10°` and `-14°`. Positive slant and weight changes are omitted from this pass.

Round 3 independently tests the thickness of vertical and horizontal stems:

```bash
/home/eroux/pvenvs/1/bin/pip install skia-pathops
/home/eroux/pvenvs/1/bin/python synthetic_benchmark/render_font_augmentation_audit.py \
  --preset round3
```

It applies directional contour changes of `-0.015em` and `+0.015em` to vary
vertical-stroke and horizontal-stroke thickness independently while minimizing
change along the other axis.

Round 4 fixes counter/hole winding after directional erosion and tests a
stronger approximately 30% stem-width change:

```bash
/home/eroux/pvenvs/1/bin/python synthetic_benchmark/render_font_augmentation_audit.py \
  --preset round4
```

Its directional contour values are `-0.030em` and `+0.030em`.

To produce a presentation sheet with an Uchen and an Ume font, each shown
unmodified and with five combined extreme variants:

```bash
/home/eroux/pvenvs/1/bin/python synthetic_benchmark/render_font_augmentation_demo.py
```

This writes `out/font_augmentation_demo/font_augmentation_demo.jpg` and a
provenance/QC manifest. The compact demo omits the fourth content line from the
review probe text.

Audit and demo renders also run raster QC against each font's baseline. Variants
are rejected when they lose too much ink density, lose enclosed counters
(including black-filled holes), or fragment into substantially more connected
components. Combined transforms apply directional stroke changes before
FontForge width/slant operations to preserve counter topology.

After review, enable the same policy directly in page generation:

```bash
/home/eroux/pvenvs/1/bin/python synthetic_benchmark/render_batches.py \
  synthetic_benchmark/out/render_plan.parquet \
  --out-dir synthetic_benchmark/out/dataset \
  --font-augmentation-variants 20 \
  --jobs 4
```

`--font-augmentation-variants N` is disabled by default and capped at 12. When
enabled, the renderer generates and validates `N` deterministic combined
variants for every source font represented in the remaining render-plan rows.
Pages for that font are distributed stably across the pool and LuaLaTeX batches
are split by variant. Font pools are prepared in parallel with
`--font-augmentation-workers` (default 4).
The approved policy samples width from `0.80–1.20`, uses a near-upright-weighted
negative slant distribution ending at `-14°`, and applies conservative
directional stroke changes to half of variants.

Every validated variant has an immutable name derived from the source-font
SHA-256, TTC face index, and complete augmentation specification. Validated
fonts and JSON sidecars are read from or written to
`s3://bec.bdrc.io/synthetic/font_cache/` by default; override with
`--font-augmentation-s3-cache-uri`, or pass an empty value to disable remote
caching. If every bounded attempt for a font fails safety validation, a
policy-keyed negative decision is also cached. Identical future runs immediately
use the original font, while a changed policy version, seed, requested count, or
probe set can evaluate it again.

Generated fonts and QC probe renders live temporarily under
`OUT_DIR/.font_augmentation_cache/` and are removed in a `finally` block after
rendering. Pass `--keep-font-augmentation-cache` for debugging. The persistent
`OUT_DIR/font_augmentation_manifest.json`, catalog fragments, and alignment
metadata retain each variant id and its exact operation labels, so page
provenance does not depend on keeping the TTF files. Re-running with the same
source fonts and `--font-augmentation-seed` regenerates the same pools.

Before a variant enters a page-rendering pool it must reopen as a valid font,
avoid new hard shaping failures on the Tibetan probe set, render successfully,
and pass the ink-density/counter/component raster checks. Affine-sensitive
placement heuristics remain part of manual review rather than the runtime gate.

## 2c. Review ink, paper, noise, and local spatial augmentation

Document-level augmentation is also kept outside dataset rendering until its
strengths and probabilities have been reviewed. With Augraphy installed in the
project environment, render the current mild/medium/strong grid with:

```bash
/home/eroux/pvenvs/1/bin/pip install augraphy
/home/eroux/pvenvs/1/bin/python \
  synthetic_benchmark/render_document_augmentation_audit.py
```

The default inputs are one Uchen and one Ume benchmark page. Outputs under
`out/document_augmentation_review/` include full JPEG samples, a reproducibility
manifest, and three contact sheets per source:

- local ink: bleed-through, dirty drum, dithering, ink bleed, rare hollowing,
  letterpress, and line degradation;
- paper/photometric noise: paper color, low-light noise, noise texture, and
  subtle noise;
- local spatial/optical effects: InkShifter, folding, and Gaussian blur.

Every row compares the source with mild, medium, and deliberately strong boundary
values. The script downsizes only these review renders to 1400 px by default; use
`--review-width 0` to inspect the original resolution. The strong column is not a
proposed production distribution. In particular, hollowing, dithering, strong
dirty-drum marks, strong low-light noise, and multi-fold samples are included to
make rejection boundaries visible.

The reviewed production policy is enabled explicitly during page rendering:

```bash
/home/eroux/pvenvs/1/bin/python synthetic_benchmark/render_batches.py \
  synthetic_benchmark/out/render_plan.parquet \
  --out-dir synthetic_benchmark/out/dataset \
  --document-augmentation \
  --document-augmentation-rate 0.90 \
  --jobs 4
```

Assignments are deterministic and balanced separately for each source font face:
60% use one of the manually selected real paper backgrounds, 30% use synthetic
paper color or page noise, and 10% remain clean white. These three paper sources
are exclusive. The selected source list is committed in
`data/paper_backgrounds.csv`; local files are used when available, otherwise
assigned sources are cached from S3.

The manifest also records luminance statistics and a light/medium/dark tier. Small
or visually fragile text is assigned only compatible paper: low effective text
resolution and lightening effects such as letterpress use light backgrounds,
while dark backgrounds are reserved for larger text. After LuaLaTeX reveals the
actual line count, dense pages are deterministically switched to a light
same-mode fallback when necessary.

Independently, the closest integer to 90% of each font's planned images receives
exactly one ink appearance effect. The weighted pool includes ink bleed,
letterpress, bleed-through, dirty drum, dithering, and rare weak Hollow.
Paper color and page noise are not applied to real-background pages. LowLightNoise and
LinesDegradation are excluded: the latter targets long straight rules and table
borders, so it had no meaningful target on the pecha pages.

InkShifter (8%), folding (3%), and blur (10%) are assigned independently per
font. Blur has only mild and medium levels. Geometric augmentation assigns
rotation to 70% of each font's pages: 60% use the typical central range up to
about 0.9 degrees, and 10% use the broader empirical tail from 0.9 to 2.4
degrees. TPS baseline curvature is applied to 30% of pages, all selected from
the rotated pages to retain the relationship seen in ldv1. Of all pages, 20%
use typical maximum vertical displacement up to 3.04% of image height and 10%
use the stronger 3.04–5.26% range.

TPS uses five correlated vertical-only control points at the empirical
horizontal positions, plus four fixed identity corner anchors for stable page
edges. The sampled corner anchors are not treated as distortion values.
Override the defaults with `--rotation-rate`, `--rotation-high-rate`,
`--tps-rate`, and `--tps-high-rate`; high rates are shares of all images and
cannot exceed their corresponding total rates.

The full deterministic assignment is written to
`OUT_DIR/document_augmentation_manifest.json`, and per-image effects and
geometric parameters are retained in checkpoint/alignment metadata. Augmented
runs preserve the selected paper mode: grayscale and RGB backgrounds are
written as matching JPEG modes, while bilevel backgrounds are written as Group
4 TIFF. Runs without `--document-augmentation` retain grayscale JPEG output.
Output widths and JPEG quality are recorded per image and summarized in
`OUT_DIR/image_output_manifest.json`. Use `--image-width-px` for a fixed-width
run, or override the range and low-quality share with
`--min-image-width-px`, `--max-image-width-px`, and
`--low-jpeg-quality-rate`.

## 3. Render Pecha Image/Alignment Pairs

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
pdftoppm -jpeg -jpegopt quality=95 -gray -scale-to-x 3500 -scale-to-y -1
```

Without document augmentation, final JPEGs are explicitly converted to grayscale
(`L`). Each page is resized to its assigned 1800–3500 px width before final
encoding. With augmentation, the intermediate starts on white and ink is composited
onto the selected real, synthetic, or clean paper before TPS, rotation, and blur.
Real backgrounds are stretched to the exact output dimensions rather than
cropped when their aspect ratio differs.

Line spacing is also deterministic but nonuniform: ordinary pages use a
triangular baseline factor from 1.18 to 1.32 times the rendered font size. Pages
whose source text contains five-, six-, or seven-plus-codepoint stacks receive
progressively larger leading; a LuaTeX glyph-offset check supplies an additional
safety floor. The base font scale is 0.65 for a condensed majority, while a
balanced 10% sparse subset receives a 1.20–1.35 multiplier. This prevents
superscript/subscript collisions without increasing ordinary line gaps globally
and retains controlled page-density variation.

Pecha page defaults:

```text
page_height = 74 mm
page_width = 4 * page_height = 296 mm
left/right margins = 20 mm
top/bottom margins = 16 mm
font scale = 0.65 * font_size_pt
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

