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

Usage:
    python src/vectors.py --acts data/acts.pt --out data/vectors.pt
"""

from __future__ import annotations

import argparse
import json
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


# ── Splitting ─────────────────────────────────────────────────────────────────

def holdout_split(n: int, n_holdout: int = 6, seed: int = 42):
    """Fixed-seed train/holdout index split."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    return idx[n_holdout:], idx[:n_holdout]


def evaluate_layers(aligned, misaligned, n_holdout=6, seed=42):
    """Re-derive vectors from the training split, score held-out pairs.

    Returns dict with per-layer held-out AUROC, separation, and the fitted vectors.
    """
    n, n_layers, _ = aligned.shape
    train_idx, hold_idx = holdout_split(n, n_holdout, seed)

    # Vectors from TRAINING pairs only
    v_train = diff_of_means(aligned[train_idx], misaligned[train_idx])

    hold_a = project(aligned[hold_idx], v_train).numpy()      # [n_hold, n_layers]
    hold_m = project(misaligned[hold_idx], v_train).numpy()

    aurocs = np.array([auroc(hold_m[:, L], hold_a[:, L]) for L in range(n_layers)])
    seps = (hold_m - hold_a).mean(0)

    return {
        "vectors_train": v_train,
        "auroc": aurocs,
        "separation": seps,
        "holdout_aligned": hold_a,
        "holdout_misaligned": hold_m,
        "train_idx": train_idx,
        "holdout_idx": hold_idx,
        "best_layer": int(np.nanargmax(aurocs)),
    }


def leave_one_out(aligned, misaligned):
    """LOO separation per layer — an alternative to a single holdout split.

    More stable than one 6-pair holdout at this dataset size, but slower and
    doesn't give a clean AUROC. Useful as a cross-check on layer choice: if LOO
    and the holdout split disagree about the best layer, the choice is not robust
    and that is worth saying out loud rather than picking whichever looks better.
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
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    blob = torch.load(args.acts)
    A, Mis = blob["aligned"], blob["misaligned"]
    cfg = blob["config"]
    n, n_layers, d_model = A.shape
    print(f"{n} pairs, {n_layers} layers, d_model={d_model}")
    print(f"model: {cfg['model_key']} -> {cfg['paper_flagship']}\n")

    res = evaluate_layers(A, Mis, args.n_holdout, args.seed)
    loo = leave_one_out(A, Mis)

    best = res["best_layer"]
    best_loo = int(np.nanargmax(loo))

    print(f"{'layer':>6}{'AUROC':>9}{'sep':>9}{'LOO sep':>10}")
    print("-" * 34)
    for L in range(n_layers):
        mark = "  <- best" if L == best else ""
        print(f"{L:>6}{res['auroc'][L]:>9.3f}{res['separation'][L]:>9.3f}"
              f"{loo[L]:>10.3f}{mark}")

    print(f"\nbest layer (held-out AUROC): {best}  AUROC={res['auroc'][best]:.3f}")
    print(f"best layer (LOO separation): {best_loo}")
    if best != best_loo:
        print("\n⚠  The two selection methods disagree. Layer choice is not robust at\n"
              "   this dataset size. Report both and say so rather than picking the\n"
              "   flattering one.")

    # Full-data vectors for deployment in the monitor; train-split vectors are
    # what the AUROC number describes. Keep both and be clear which is which.
    v_full = diff_of_means(A, Mis)

    torch.save({
        "vectors_full": v_full,
        "vectors_train": res["vectors_train"],
        "auroc": res["auroc"],
        "separation": res["separation"],
        "loo_separation": loo,
        "best_layer": best,
        "best_layer_loo": best_loo,
        "holdout_idx": res["holdout_idx"],
        "train_idx": res["train_idx"],
        "holdout_aligned": res["holdout_aligned"],
        "holdout_misaligned": res["holdout_misaligned"],
        "config": {**cfg, "n_holdout": args.n_holdout, "split_seed": args.seed},
    }, args.out)

    print(f"\n-> {args.out}")
    print("\nvectors_full  = fitted on all pairs, use in the monitor")
    print("vectors_train = fitted on train split, what the AUROC describes")


if __name__ == "__main__":
    main()
