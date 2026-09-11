<proposed_plan>
# Repair Scratch papers 15–25

## Goal
Keep all committed code. Do not rework index 0–14.
Index 15–25 has reset state and must be repaired strictly one paper at a time.

The goal is not merely to make each paper runnable. The goal is to turn its algorithm into reusable Scratch particles.

## Per-paper loop

1. Read the current Scratch recipe/blocks first, then read the paper, formal config and maintained legacy implementation.

2. Build a temporary mapping for the current paper:

   paper formula / meaningful Algorithm step
   → current Scratch block

   Classify every item:
   - KEEP: already a correct reusable particle
   - MODIFY: useful but incomplete
   - REPLACE: wrong, duplicate, legacy wrapper or giant block
   - MISSING: no corresponding Scratch particle

3. Repair the current paper.

   A valid particle must:
   - represent one meaningful formula or algorithm step;
   - expose reusable input slots / parameters / output;
   - not depend on paper name, recipe name or fixed Context keys;
   - reuse an existing mathematically equivalent Scratch block when one exists;
   - not call an entire legacy algorithm as its implementation.

   If several meaningful paper steps are hidden inside one block, split it.
   Old giant blocks may remain hidden for compatibility, but the Paper Recipe must not use them.

4. Before running tests, inspect the mapping again.

   Every core paper formula/Algorithm step must have a reasonable Scratch particle.
   A paper is not considered repaired merely because its Recipe runs.

5. Only after particleization is complete run:
   - recipe validate;
   - focused formula/block checks;
   - runtime-limited or fixture structural execution;
   - WebUI recipe load/Validate.

   These tests prove executability only; they do not replace the particleization check.

6. Mark the paper READY only after steps 1–5 are complete.
   Then increment current_index and only then inspect the next paper.

Use GCE, JoCoR, VolMinNet and CWD from index 0–14 as quality references:
- GCE: formula chain;
- JoCoR: multi-model interaction and selection;
- VolMinNet: separate objective terms then composition;
- CWD: statistics/state stages separated from batch training.

After index 25, run the full Scratch regression suite once.
</proposed_plan>