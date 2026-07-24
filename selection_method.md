# Source-text selection

The benchmark should contain difficult Tibetan stacks without filling many pages
with the same rare forms. Text is therefore selected for **coverage**, not just
for a high rarity score.

For every font face, the planner remembers which rare stacks have already
appeared. When filling a difficult-text slot, it examines a bounded set of
compatible chunks and favors chunks that introduce stacks not yet selected for
that font. Once a chunk is chosen, its rare stacks are added to that font's
covered set.

The ranking balances:

- the number of newly covered rare stacks;
- the rarity of those new stacks;
- the requested low- or high-repetition behavior;
- the chunk's overall difficulty.

The annotation marks `༵` and `༷` do not create separate rarity identities.
Their corpus counts and coverage are merged with the corresponding unmarked
stack, so heavily annotated passages cannot consume the rare-stack budget merely
by producing many marked variants.

Coverage is tracked separately for each font because a stack rendered by one
font does not test another font. The candidate must also respect font support,
dataset split boundaries, and reuse limits.

Half of each font's pages still use normal text with no stacks above the rarity
threshold. The other half uses this coverage-driven difficult selection. Of all
slots, 90% prefer low repetition and 10% deliberately preserve repetitive text.
If a font cannot render enough rare-stack chunks, its difficult slots fall back
to the most structurally complex compatible common text.

At present, novelty is measured over **rare stacks**. A possible extension is to
reward every unseen stack while giving unseen rare stacks a larger weight. That
would improve broad stack coverage without allowing common stacks to overwhelm
the difficult-text objective.
