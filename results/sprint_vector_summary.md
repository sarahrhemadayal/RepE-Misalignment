
## Sprint result — reading vector (2026-08-27)

**Model:** qwen3-8b -> Qwen3-235B-A22B [same-family]
**Dataset:** 30 deception pairs, 6 held out per split (seed 42)
**Method:** difference-of-means over last-8-token mean-pooled resid_post; best layer
chosen as argmax of held-out AUROC averaged over 20 random holdout splits
(fixes a tie-break bug in the original single-split selection — see notebook section 3).

| metric | value |
|---|---|
| best layer (repeated-holdout AUROC) | 8 |
| repeated-holdout AUROC (mean +/- std) | 0.978 +/- 0.035 |
| held-out AUROC, canonical split | 0.889 |
| Cohen's d at best layer | 1.72 |
| best layer, old single-split method | 7 |
| best layer, raw LOO (depth-confounded) | 35 |

### Caveat (for the slide, verbatim)
Correlational separation on held-out synthetic pairs. This is NOT causal proof
that this direction IS deception, and it is untested off-distribution. Establishing
causality requires steering (add the vector — does misaligned output increase?) and
ablation (remove it — does it decrease?), neither of which is done here.

### Further limits
- 30 pairs is small; a single 6-pair holdout is noisy enough that AUROC
  ties at the ceiling across many layers — that is why layer selection is now averaged
  over 20 holdout splits rather than read from one.
- All pairs are synthetic and written by one person — shared authorial style is a
  plausible confound; extraction now mean-pools the last 8 reasoning tokens rather than
  the single final token specifically to reduce (not eliminate) this at the token-identity level.
- qwen3-8b is a small descendant of Qwen3-235B-A22B, not that model.
- Single concept (deception). shutdown_avoidance and power_seeking not yet built.
