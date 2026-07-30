# Source-text selection

The benchmark should contain difficult Tibetan stacks without filling many pages
with the same rare forms. Text is therefore selected for **coverage**, not just
for a high rarity score.

For every font face, the planner remembers which rare stacks have already
appeared. When filling a difficult-text slot, it examines a bounded set of
compatible chunks and favors chunks that introduce stacks not yet selected for
that font. Once a chunk is chosen, its rare stacks are added to that font's
covered set.

Before ranking, chunks must pass hard text-quality filters:

- repetition score `< 0.50` (dominant syllables, adjacent duplicates, and
  repeated n-grams);
- normalized Zstd compression ratio `>= 0.18` (level-3 compressed UTF-8 bytes
  divided by raw UTF-8 bytes).

The Zstd ratio is an information-density / redundancy score. It correlates
strongly with the repetition score, but also flags formulaic passages whose
nouns change while the surrounding template stays the same. Both scores are
stored on every plan row.

The ranking then balances:

- the number of newly covered rare stacks;
- the rarity of those new stacks;
- preference for lower repetition among remaining eligible chunks;
- the chunk's overall difficulty.

The annotation marks `༵` and `༷` do not create separate rarity identities.
Their corpus counts and coverage are merged with the corresponding unmarked
stack, so heavily annotated passages cannot consume the rare-stack budget merely
by producing many marked variants.

Coverage is tracked separately for each font because a stack rendered by one
font does not test another font. The candidate must also respect font support,
dataset split boundaries, and reuse limits.

Half of each font's pages still use normal text with no stacks above the rarity
threshold. The other half uses this coverage-driven difficult selection. If a
font cannot render enough rare-stack chunks, its difficult slots fall back to
the most structurally complex compatible common text.

At present, novelty is measured over **rare stacks**. A possible extension is to
reward every unseen stack while giving unseen rare stacks a larger weight. That
would improve broad stack coverage without allowing common stacks to overwhelm
the difficult-text objective.

## Reading the excluded-stack coverage figure

Compare distinct letter stacks in the chunk corpus with those that appear in the
selected plan, but always weight by BoCorpus occurrence counts. Stacks containing
U+0F35 or U+0F37 are ignored entirely. On the current ~45.7k-page
density-filtered plan:

- corpus distinct letter stacks: 9,752
- selected: 4,826
- excluded: 4,926 (~0.009% of corpus occurrence mass)

Classify each excluded stack:

1. **font_support** — unsupported by every eligible font (92 stacks; mass 536).
2. **chunk_exclusion** — only attested in chunks removed by the text policy
   (51 stacks; mass 67). Of those, density-only and repetition-and-density
   account for all cases; repetition-only removes almost no unique stacks
   (one chunk fails repetition alone in this run). Attribute these stacks to
   BoCorpus `collection` via `bocorpus_row`.
3. **selection** — eligible and font-supported, but not chosen under the page /
   quota / rare-stack novelty budget (4,783 stacks; mass 25,853).

Regenerate the tables with:

```bash
PYTHONPATH=synthetic_benchmark python \
  synthetic_benchmark/create_excluded_stacks_report.py \
  --plan PATH/TO/render_plan.parquet \
  --output-dir reports/excluded_stacks_report
```
