# From coverage matrix to pecha pages

*Part 2 of a series on building a synthetic OCR benchmark for Tibetan — work supported by a [Khyentse Foundation](https://khyentsefoundation.org/) grant to improve Tibetan OCR at BDRC / OpenPecha.*

[Part 1](01-font-coverage-before-synthetic-ocr.md) asked which fonts can draw which stacks. This post is the short next step: **turn accepted `(font, text)` pairs into pecha-shaped JPEG pages with line-accurate transcriptions**, using LuaLaTeX.

![Example synthetic uchen pecha page (Aathup)](assets/02-rendering/pecha_uchen_example.jpg)

---

## Pipeline in one diagram

```text
BoCorpus chunks
    + stack_support.parquet   (from coverage_report)
    → build_render_plan.py    (reject unsupported stacks)
    → render_batches.py
         LuaLaTeX / fontspec / HarfBuzz
         → PDF pages
         → pdftoppm (mixed JPEG/TIFF, width 1800–3500 px)
         → alignment parquet (transcription with real line breaks)
```

The planner never asks a font to render a chunk that contains an unsupported stack. That is the whole point of part 1.

---

## Why LuaLaTeX?

We need the same OpenType Tibetan shaping that HarfBuzz used in the coverage gate, on real multi-line pages, with a known font file (including TTC face index). LuaLaTeX + `fontspec` gives us that, plus:

- pecha geometry (`paperwidth ≈ 4 × paperheight`, wide margins)
- per-syllable markers so we can rebuild the ground-truth text with the **same** line breaks the PDF produced
- batching many pages per font in one TeX run

Defaults look like a digital pecha strip: height 74 mm, width 296 mm, ~20 mm
side margins, and font size at `0.65 ×` the catalog’s normalized
`font_size_pt`. This keeps 90% of pages reasonably condensed; a deterministic
10% sparse subset receives a `1.20–1.35×` size multiplier. Variable leading and
extra clearance for tall stacks add independent density variation.
Every other exported page can get a `༄༅། །` prefix, matching common volume
openings.

![Uchen page crop (Aathup)](assets/02-rendering/uchen_none_crop.jpg)

![Ume cursive page crop (Thonmi Khyug)](assets/02-rendering/ume_none_crop.jpg)

---

## What a render plan row is

Each row is roughly: *one intended image* = one BoCorpus chunk + one font face + split label (train/val/test) + script taxonomy fields. The planner balances volume across the catalog’s `8 categories` (Zabma, Parma, Druma, Tsugma, …) while keeping uchen and ume volumes planned separately for later training mixes.

Source text is balanced independently for every font face, before shorthand or
other text augmentation:

- **50% normal:** compatible prose chunks containing no stacks above the corpus
  rarity threshold;
- **50% difficult:** compatible chunks prioritizing rare stacks and broader
  difficult-stack coverage.

Some fonts cannot render enough globally rare stacks to fill their difficult
half. In that case, the planner uses the most structurally complex compatible
common chunks for that font instead. The policy tier, actual rarity tier, and
fallback basis are recorded separately in
`source_text_difficulty_tier`, `source_text_rarity_tier`, and
`source_text_difficulty_basis`. Counts can differ by one when a font receives an
odd number of pages.

Heavy repetition is removed before selection. Every candidate chunk must have
`source_text_repetition_score < 0.50`. The score is the maximum of the
dominant-syllable fraction, adjacent-duplicate fraction, repeated-trigram
fraction, and a down-weighted repeated-bigram fraction. There is no longer a
rewarded-repetition slot that prefers repetitive text.

Information density is measured separately with a normalized Zstd score:
`source_text_zstd_compression_ratio` = compressed UTF-8 bytes / raw UTF-8 bytes
at fixed Zstd level 3. Lower ratios mean more redundancy (lists, formulaic
passages, long repeated phrases). The current production filter also requires
`source_text_zstd_compression_ratio >= 0.18`. The two metrics are strongly
correlated, but Zstd still catches some template-like text that the syllable
repetition score alone would keep.

Rare text is selected for **coverage**, rather than repeatedly taking whichever
chunks have the highest global difficulty score. For each font face, the planner
remembers the rare stacks already selected. For a difficult slot it examines up
to 96 compatible candidates and prefers chunks that introduce previously unseen
rare stacks for that font.

The ranking gives most weight to the number and rarity of newly covered stacks,
while retaining smaller terms for lower repetition and overall chunk difficulty.
After selection, the chunk's rare stacks enter that font's covered set, changing
the ranking for the next page. This greedy update spreads difficult examples
across distinct forms instead of concentrating pages around the same few rare
stacks. Coverage gains, newly introduced stack counts, repetition scores, and
Zstd ratios are stored in the render plan for auditing. A concise description is
available in [`selection_method.md`](../selection_method.md).

For rarity and novelty only, stacks containing `༵` or `༷` are folded into their
unmarked base stack. The marks remain in the rendered text, but they no longer
manufacture hundreds of nominally rare variants at the expense of mantra and
other structurally complex stacks.

### What the coverage figure leaves out

The headline stack-coverage numbers compare distinct letter stacks in the chunk
corpus against those that appear in the selected plan. Stacks containing the
annotation marks U+0F35 / U+0F37 are ignored entirely (neither in the corpus
denominator nor in the selected/excluded counts). On the current ~45.7k-page
plan that is **4,934 / 9,752** selected (~50.6%). Counting distinct forms alone
is misleading: weighted by BoCorpus occurrence frequency, the **excluded** stacks
are only **~0.008%** of corpus mass (24,406 / 290M occurrences). Everyday Tibetan
is covered; the missing half of the unique list is almost entirely the long
rare tail.

We classify every excluded stack into one of three reason groups:

| Reason group | Meaning | Distinct stacks | Occurrence mass |
| --- | --- | ---: | ---: |
| **selection** | Font-supported and present in text-policy-eligible chunks, but not chosen by coverage/quota selection | 4,676 | 23,805 |
| **font_support** | Never supported by any eligible benchmark font under `stack_support.parquet` | 91 | 534 |
| **chunk_exclusion** | Appears in BoCorpus chunks, but only in chunks removed by repetition and/or Zstd density filters | 51 | 67 |

**Font support vs chunk exclusion.** True hard failures of the coverage gate are
small: 91 stacks, mostly pathological Sanskrit-style piles (for example `ྈྐ`,
`ྉྤ`, doubled *anusvāra* / *candrabindu* forms). They account for ~2% of
excluded occurrence mass and ~0.0002% of total corpus mass.

**Chunk exclusion is not “repetition only.”** Of 415,115 chunks, the text policy
keeps 359,258 and drops 55,857. Almost all removals involve the Zstd density
floor: 42,363 fail both filters, 13,493 fail density alone, and only **1** chunk
fails repetition alone. Correspondingly, the 51 stacks that vanish entirely
because of chunk filtering break down as **33 density-only** and **18
repetition-and-density** — none are repetition-only. Their BoCorpus sources are
concentrated in Derge Tengyur / Kangyur, Bon Kangyur, Rinchen Terdzö, and
འབྲི་ལུགས་བང་མཛོད་སྐོར་ལྔ།; they are ultra-rare forms (often 1–6 corpus
occurrences) that happen to sit only inside highly redundant passages.

**Selection dominates the distinct-count gap.** The remaining 4,676 excluded
stacks are eligible and renderable, but the 45.7k-page budget (with per-font
coverage greedy over *rare* stacks, plus category quotas) never picks a chunk
that contains them. Most are scarce in the eligible pool (median once). Novelty
only tracks rarity ≥ 0.8, so frequency-"common" misses such as `རྨྱི` (663
occurrences) get no coverage pressure. Most of that selection gap's mass sits in
Derge Tengyur, with smaller contributions from Derge Kangyur,
འབྲི་ལུགས་བང་མཛོད་སྐོར་ལྔ།, the Pagen Project, and Rinchen Terdzö. Closing more of
this gap means changing the novelty / fill pass, not relaxing text-quality
filters.

Machine-readable detail for this plan lives under
`reports/v3_density018/excluded_stacks_report/`
(`excluded_stacks_summary.json`, per-stack CSV with collection attribution).

At render time, chunks that overflow a physical page contribute only their **first** page to the benchmark; short underfull pages can be merged with the next chunk and re-rendered. Failures are logged per batch; successful batches checkpoint so long runs are resumable.

---

## Output shape

```text
images/.../0001.jpg          grayscale pecha JPEG
alignments/.../*_ptt.parquet transcription + font + script_8 + source span
```

That is enough for OCR training and for BUDA-style alignment catalogs. Linguistic tricks (shorthands) are optional and off by default — that is [part 3](03-shorthand-augmentations.md).

---

## Open source

```bash
python synthetic_benchmark/build_bocorpus_chunks.py
python synthetic_benchmark/build_render_plan.py \
  --support-parquet coverage_report/out/stack_support.parquet \
  --target-images 500000
python synthetic_benchmark/render_batches.py \
  synthetic_benchmark/out/render_plan.parquet \
  --out-dir synthetic_benchmark/out/dataset \
  --jobs 4
```

Code: [`synthetic_benchmark/`](../synthetic_benchmark/) in [buda-base/synthetic-ocr-benchmark-tools](https://github.com/buda-base/synthetic-ocr-benchmark-tools).

*Next: [injecting Tibetan shorthands without breaking font coverage](03-shorthand-augmentations.md).*
