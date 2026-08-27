"""
vectors.py — Reading-vector extraction and layer selection.

Builds per-layer difference-of-means directions from contrastive activations,
then evaluates each layer on HELD-OUT pairs and reports AUROC.

    v_L = normalise( mean(misaligned_L) - mean(aligned_L) )

The held-out discipline matters more than it looks. With 30 pairs, choosing the
best layer by in-sample separation would overfit badly — the "best" layer would
partly be the one where the training noise happened to line up. So the vector is
re-derived from the training split only, and the held-out pairs are scored
against that.

Layer selection: single split vs. repeated (2026-08-27 fix)
------------------------------------------------------------
A single 6-pair holdout quantizes AUROC to multiples of 1/(n_pos*n_neg) (with
3 aligned / 3 misaligned held out, steps of 1/9 ~= 0.111). When separation is
this dataset's biggest problem — most layers hit a perfect 1.000 — `np.argmax`
silently returns the FIRST tied layer, which is why layer 0 kept winning: not
because it is genuinely the best layer, but because ties break toward the front
of the array. `repeated_holdout_auroc` reruns the single-split evaluation over
many random holdout partitions and averages, which de-quantizes the statistic
so the "best layer" reflects where separation is consistently strong rather
than an artifact of tie order. `main()` now selects the reported best layer
this way; the original single-split `evaluate_layers` is kept (and still used,
at the default seed, to produce the canonical held-out sample that
`monitor.py`/`choose_tau` scores against) so nothing downstream needs to change.

Usage:
    python src/vectors.py --acts data/acts.pt --out data/vectors.pt
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch


# ── Core ──────────────────────────────────────────────────────────────────────

def diff_of_means(aligned: torch.Tensor, misaligned: torch.Tensor) -> torch.Tensor:
    """Per-layer unit-normalised difference-of-means direction.

    aligned, misaligned: [n_pairs, n_layers, d_model]
    returns:             [n_layers, d_model]
    """
    v = misaligned.mean(0) - aligned.mean(0)
    return v / v.norm(dim=-1, keepdim=True)


def project(acts: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Dot-product score of activations against per-layer directions.

    acts:    [n_pairs, n_layers, d_model]
    vectors: [n_layers, d_model]
    returns: [n_pairs, n_layers]
    """
    return (acts * vectors.unsqueeze(0)).sum(-1)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U identity — no sklearn dependency, and exact.

    Equivalent to P(random positive scores above random negative), with ties
    counted as half.
    """
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[:n_pos].sum()
    return (r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def cohens_d(pos: np.ndarray, neg: np.ndarray) -> float:
    """Standardized effect size: (mean(pos) - mean(neg)) / pooled std.

    Raw dot-product "separation" is NOT comparable across layers: the residual
    stream's norm grows hugely with depth (observed here: ~0.3 at layer 0 to
    ~130+ at layer 35), so later layers post bigger raw separation numbers
    almost by construction, independent of whether the concept is any cleaner
    there. Dividing by the pooled within-class std removes that scale, giving
    a number that is actually meaningful to compare layer-to-layer.
    """
    n_pos, n_neg = len(pos), len(neg)
    if n_pos < 2 or n_neg < 2:
        return float("nan")
    var_pos, var_neg = pos.var(ddof=1), neg.var(ddof=1)
    pooled_std = np.sqrt(((n_pos - 1) * var_pos + (n_neg - 1) * var_neg) / (n_pos + n_neg - 2))
    if pooled_std == 0:
        return float("nan")
    return float((pos.mean() - neg.mean()) / pooled_std)


# ── Splitting ─────────────────────────────────────────────────────────────────

def holdout_split(n: int, n_holdout: int = 6, seed: int = 42):
    """Fixed-seed train/holdout index split."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    return idx[n_holdout:], idx[:n_holdout]


def evaluate_layers(aligned, misaligned, n_holdout=6, seed=42):
    """Re-derive vectors from the training split, score held-out pairs.

    Returns dict with per-layer held-out AUROC, separation, standardized
    effect size (Cohen's d), and the fitted vectors.
    """
    n, n_layers, _ = aligned.shape
    train_idx, hold_idx = holdout_split(n, n_holdout, seed)

    # Vectors from TRAINING pairs only
    v_train = diff_of_means(aligned[train_idx], misaligned[train_idx])

    hold_a = project(aligned[hold_idx], v_train).numpy()      # [n_hold, n_layers]
    hold_m = project(misaligned[hold_idx], v_train).numpy()

    aurocs = np.array([auroc(hold_m[:, L], hold_a[:, L]) for L in range(n_layers)])
    seps = (hold_m - hold_a).mean(0)
    ds = np.array([cohens_d(hold_m[:, L], hold_a[:, L]) for L in range(n_layers)])

    return {
        "vectors_train": v_train,
        "auroc": aurocs,
        "separation": seps,
        "cohens_d": ds,
        "holdout_aligned": hold_a,
        "holdout_misaligned": hold_m,
        "train_idx": train_idx,
        "holdout_idx": hold_idx,
        "best_layer": int(np.nanargmax(aurocs)),
    }


def repeated_holdout_auroc(aligned, misaligned, n_holdout=6, n_repeats=20, seed=42):
    """Average held-out AUROC over many random holdout splits.

    See module docstring: this exists because a single 6-pair holdout produces
    so many tied-at-1.000 layers that argmax's tie-break (first index) was
    silently deciding "best layer", not the data. Averaging over `n_repeats`
    independent splits de-quantizes AUROC and makes the winner reflect
    consistent separation instead. `win_counts` reports how often each layer
    won its own split, as a robustness check on the averaged pick.
    """
    n, n_layers, _ = aligned.shape
    all_auroc = np.zeros((n_repeats, n_layers))
    winners = []
    for r in range(n_repeats):
        res = evaluate_layers(aligned, misaligned, n_holdout=n_holdout, seed=seed + r)
        all_auroc[r] = res["auroc"]
        winners.append(res["best_layer"])

    mean_auroc = np.nanmean(all_auroc, axis=0)
    std_auroc = np.nanstd(all_auroc, axis=0)
    best_layer = int(np.nanargmax(mean_auroc))

    return {
        "mean_auroc": mean_auroc,
        "std_auroc": std_auroc,
        "best_layer": best_layer,
        "win_counts": Counter(winners),
        "n_repeats": n_repeats,
    }


def leave_one_out(aligned, misaligned):
    """LOO separation per layer — an alternative to a single holdout split.

    More stable than one 6-pair holdout at this dataset size, but slower and
    doesn't give a clean AUROC. Useful as a cross-check on layer choice: if LOO
    and the holdout split disagree about the best layer, the choice is not robust
    and that is worth saying out loud rather than picking whichever looks better.

    Caveat: like `separation` above, this is a raw dot product and inherits the
    same depth-scaling confound — it will trivially favour the deepest layer
    almost regardless of concept quality. Treat it as a sanity check on
    direction, not as a layer-selection criterion on its own.
    """
    n, n_layers, _ = aligned.shape
    sep = np.zeros(n_layers)
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        v = diff_of_means(aligned[idx], misaligned[idx])
        sep += ((misaligned[i] * v).sum(-1) - (aligned[i] * v).sum(-1)).numpy()
    return sep / n


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Build reading vectors + evaluate layers")
    p.add_argument("--acts", default="data/acts.pt")
    p.add_argument("--out", default="data/vectors.pt")
    p.add_argument("--n-holdout", type=int, default=6)
    p.add_argument("--n-repeats", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    blob = torch.load(args.acts)
    A, Mis = blob["aligned"], blob["misaligned"]
    cfg = blob["config"]
    n, n_layers, d_model = A.shape
    print(f"{n} pairs, {n_layers} layers, d_model={d_model}")
    print(f"model: {cfg['model_key']} -> {cfg['paper_flagship']}\n")

    # Canonical single split (seed) — used for tau selection downstream and
    # for the diagnostic Cohen's d column.
    res = evaluate_layers(A, Mis, args.n_holdout, args.seed)
    loo = leave_one_out(A, Mis)

    # Robust layer pick — averaged over many holdout splits (the fix).
    rep = repeated_holdout_auroc(A, Mis, args.n_holdout, args.n_repeats, args.seed)

    best_single = res["best_layer"]
    best_loo = int(np.nanargmax(loo))
    best_layer = rep["best_layer"]  # <- what gets used downstream

    dcol = "d"
    print(f"{'layer':>6}{'AUROC(1)':>10}{'AUROC(rep)':>12}{'+/-':>7}{dcol:>8}{'LOO':>10}")
    print("-" * 53)
    for L in range(n_layers):
        mark = "  <- best (repeated)" if L == best_layer else ""
        print(f"{L:>6}{res['auroc'][L]:>10.3f}{rep['mean_auroc'][L]:>12.3f}"
              f"{rep['std_auroc'][L]:>7.3f}{res['cohens_d'][L]:>8.2f}{loo[L]:>10.3f}{mark}")

    print(f"\nbest layer, single split (seed={args.seed}):  {best_single}  "
          f"AUROC={res['auroc'][best_single]:.3f}  <- old method, prone to tie-break artifacts")
    print(f"best layer, repeated ({args.n_repeats} splits):    {best_layer}  "
          f"mean AUROC={rep['mean_auroc'][best_layer]:.3f} +/- {rep['std_auroc'][best_layer]:.3f}")
    print(f"best layer, LOO (raw, depth-confounded): {best_loo}")
    print(f"\nper-split win counts (top 5): {rep['win_counts'].most_common(5)}")

    if best_single != best_layer:
        print(f"\n note: single-split pick (L{best_single}) and repeated pick (L{best_layer}) "
              f"differ — this is exactly the tie-break artifact the repeated method exists to fix.")

    # Full-data vectors for deployment in the monitor; train-split vectors are
    # what the AUROC number describes. Keep both and be clear which is which.
    v_full = diff_of_means(A, Mis)

    torch.save({
        "vectors_full": v_full,
        "vectors_train": res["vectors_train"],
        "auroc": res["auroc"],                       # canonical single-split (seed), for continuity
        "separation": res["separation"],
        "cohens_d": res["cohens_d"],
        "loo_separation": loo,
        "best_layer": best_layer,                     # <- now the repeated-split pick
        "best_layer_single_split": best_single,
        "best_layer_loo": best_loo,
        "auroc_repeated_mean": rep["mean_auroc"],
        "auroc_repeated_std": rep["std_auroc"],
        "win_counts": dict(rep["win_counts"]),
        "holdout_idx": res["holdout_idx"],
        "train_idx": res["train_idx"],
        "holdout_aligned": res["holdout_aligned"],
        "holdout_misaligned": res["holdout_misaligned"],
        "config": {**cfg, "n_holdout": args.n_holdout, "n_repeats": args.n_repeats, "split_seed": args.seed},
    }, args.out)

    print(f"\n-> {args.out}")
    print("\nvectors_full  = fitted on all pairs, use in the monitor")
    print("vectors_train = fitted on train split, what the AUROC number describes")


if __name__ == "__main__":
    main()
