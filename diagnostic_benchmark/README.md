# Diagnostic Tibetan OCR Benchmark (OFAT)

A small, controlled **diagnostic** OCR benchmark: 300 rendered Tibetan page
images with exact, noise-free ground truth, built as a *scientific instrument*
to isolate how OCR character-error-rate (CER) responds to individual difficulty
axes **one factor at a time** (OFAT). This is **not** a coverage-maximizing
training set; every axis varies a single property while all other properties are
held at a fixed "center" configuration.

Ground truth (`transcription`) is the LuaLaTeX-rendered text reconstructed
line-for-line from TeX syllable markers and normalized with the **exact**
`normalize_unicode_text` (NFD + canonical reorder, then graphical folding) used
by the `bec-ocr-training` scorer, so a prediction and this reference are always
compared in the same normal form.

## Schema

`benchmark.parquet` has exactly these columns (drop-in for
`bec-ocr-training/scripts/paddle_reeval_full.py`):

`file_name, transcription, image, script, technology, benchmark_group, axis, axis_bin, axis_value`

* `image` is raw grayscale JPEG bytes; the same images are also under `images/`.
* `transcription` preserves rendered line breaks as `\n` (one per visual line),
  exactly like the main pipeline's `rebuild_rendered_transcription`, so it scores
  like-for-like against page-level predictions in the existing harness.
* `technology` = `synthetic` for every row.
* `benchmark_group` = `axis_<name>` (e.g. `axis_stack_rarity`); `axis` = `<name>`.
* `axis_bin` is an ordinal integer (0 = center/easiest); `axis_value` is the
  numeric level of the axis for that image (target rep_score, syllable count,
  font size, or the source page's rare-stack score).

`manifest.csv` carries per-image provenance and realized metrics (font, source
chunk id, seed, line count, realized `rep_score`, raster-naturalization
parameters, etc.). `provenance.json` records the full configuration.

## Fixed "center" configuration

Every image is identical on all factors except the one its axis varies:

| factor | value |
|---|---|
| font | **Jomolhari-id** (`jomolhari-id.ttf`, ps `Jomolhari-ID`), a clean Uchen Unicode face |
| script | uchen |
| rendered font size | 18.0 pt (medium bin of the font-size axis) |
| page width | 200.0 mm (height grows with line count) |
| margins | 15.0 mm left/right, 10.0 mm top/bottom |
| line-spacing factor | 1.5 |
| image width | **2000 px** (fixed for every image) |
| output | grayscale JPEG, quality 90 |
| center length | ~1200 characters/page |
| repetition | ~0 (low, char-20-gram) |
| stacks | all common (no rare stacks) for non-rarity axes |
| character set | **Tibetan Unicode only** (U+0F00–U+0FFF); no Latin digits or other-script characters in any ground truth |
| font coverage gate | every stack must be supported in `coverage_report/out/stack_support.parquet` for this font (same HarfBuzz + placement rules as the main synthetic pipeline); orphan marks outside tokenized stacks are also rejected |
| augmentation | **none** — no augraphy ink/paper, no font augmentation, no shorthands, no rotation, no TPS |
| raster processing | only the mandatory minimum "profile A" naturalization (slight sub-pixel warp, faint Gaussian blur, fine luminance noise, sparse white-ink dropout) applied to *every* image so no augmentation-difficulty variance leaks into the sweeps |
| page-head prefix | none (no `༄༅། །`) |
| seed | 1729 (everything deterministic) |

Because page **width**, **font size**, **margins**, and **line spacing** are
fixed, the number of characters per line (density) is held constant; the length
axis therefore adds *lines* (a taller page at the same pixel width and dpi)
rather than making long pages denser.

## Axes

Each axis is a sweep of bins holding all other factors at the center value.

### `axis_stack_rarity`
Increasingly rare Tibetan stacks per page. Each source page is scored by the
mean corpus rarity of its 20 rarest distinct stacks (percentile-clipped
self-information from `coverage_report/bocorpus_stacks.csv`): 0.0 for an
all-common page, approaching 1.0 for a page dominated by rare / unseen stacks.
Pages are held at the center length and binned by this score; bin 0 is the
all-common center. `axis_value` = the page's rare-stack score. Only pages whose
*every* stack is coverage-supported for the chosen font are kept.

| bin | axis_value | n | mean sylls | mean chars | mean lines | mean rep |
|---|---|---|---|---|---|---|
| 0 | 0.0 | 16 | 300.6 | 1167.5 | 12.9 | 0.0171 |
| 1 | 0.005619 | 16 | 302.4 | 1193.4 | 13.4 | 0.0056 |
| 2 | 0.055223 | 16 | 302.3 | 1196.8 | 13.1 | 0.0105 |
| 3 | 0.133834 | 16 | 322.4 | 1203.4 | 12.9 | 0.0051 |
| 4 | 0.32899 | 16 | 319.1 | 1190.1 | 12.7 | 0.0027 |

### `axis_repetition`
Increasing within-page repetition (`rep_score`, char-20-gram, same definition as
the `bec-ocr-training` loop docs), synthesized by repeating a shad-delimited
motif interleaved with unique text at fixed page length. `axis_value` = target
`rep_score`.

| bin | axis_value | n | mean sylls | mean chars | mean lines | mean rep |
|---|---|---|---|---|---|---|
| 0 | 0.0 | 16 | 305.1 | 1177.6 | 13.1 | 0.0189 |
| 1 | 0.15 | 16 | 316.8 | 1238.3 | 13.8 | 0.1071 |
| 2 | 0.3 | 16 | 329.6 | 1252.9 | 14.1 | 0.2316 |
| 3 | 0.45 | 16 | 318.8 | 1233.8 | 13.7 | 0.3232 |
| 4 | 0.6 | 16 | 328.5 | 1274.5 | 14.3 | 0.446 |

### `axis_length`
Increasing syllables-per-page at fixed characters-per-line (constant density,
constant pixel width and dpi; the page just gets taller). Starts at a short
phrase (~15 syllables) and tops out at the longest page in the BEC training
corpus `bec_mixed_elie_v3` (1005 syllables) + 20% → 1206. Intermediate bins are
log-spaced. `axis_value` = target syllables/page.

| bin | axis_value | n | mean sylls | mean chars | mean lines | mean rep |
|---|---|---|---|---|---|---|
| 0 | 15.0 | 16 | 15.0 | 57.7 | 1.0 | 0.0 |
| 1 | 45.0 | 16 | 45.0 | 173.9 | 2.3 | 0.0054 |
| 2 | 135.0 | 16 | 135.0 | 516.8 | 6.1 | 0.0127 |
| 3 | 400.0 | 16 | 400.0 | 1541.6 | 17.2 | 0.0182 |
| 4 | 1206.0 | 16 | 1206.0 | 4706.9 | 51.2 | 0.0108 |

### `axis_font_size`
Control axis for effective resolution: three rendered point sizes at the fixed
image pixel width (smaller point size = fewer pixels per glyph = lower effective
resolution). `axis_value` = font size in points.

| bin | axis_value | n | mean sylls | mean chars | mean lines | mean rep |
|---|---|---|---|---|---|---|
| 0 | 12.0 | 20 | 301.9 | 1170.4 | 8.9 | 0.0248 |
| 1 | 18.0 | 20 | 309.0 | 1181.0 | 13.1 | 0.0194 |
| 2 | 28.0 | 20 | 302.4 | 1173.5 | 20.4 | 0.007 |

## Source text

Text is sourced from OpenPecha **BoCorpus** page-sized chunks
(`synthetic_benchmark/build_bocorpus_chunks.py`), already normalized with the
repository's Botok Unicode + graphical normalization. Ground truth is then
re-normalized with the vendored `normalize_unicode_text` so it is byte-identical
to the `bec-ocr-training` scoring normal form. For the repetition axis, repetition
is synthesized by repeating shad-delimited units. Pages are accepted only when
every stack is supported under `coverage_report/out/stack_support.parquet` for
`Jomolhari-id`.

## Regenerate deterministically

From the repository root, one seeded command reproduces the whole benchmark:

```bash
python diagnostic_benchmark/build_diagnostic_benchmark.py \
  --font Jomolhari-id \
  --chunks synthetic_benchmark/out/bocorpus_chunks.parquet \
  --stacks-csv coverage_report/bocorpus_stacks.csv \
  --support-parquet coverage_report/out/stack_support.parquet \
  --out-dir diagnostic_benchmark/out \
  --seed 1729 \
  --jobs 8
```

Add `--smoke` for a few images per bin to validate the LuaLaTeX/Poppler path
before the full run.

## Rights & citation

Source texts are public-domain BoCorpus; the synthetic rendered images are also
released as public domain. Please cite the Buddhist Digital Resource Center
(BDRC), the BDRC Etext Corpus project, the OpenPecha BoCorpus, and the OpenPecha
Tibetan Fonts repository.
