"""
extract.py — Residual-stream extraction for contrastive pairs.

For each pair, runs both the aligned and misaligned prompt through the model and
caches `resid_post` at every layer, keeping a WINDOW-MEAN activation: the mean
of the last `window` token positions of the reasoning trace (default 8).

Output: data/acts.pt containing two tensors of shape [n_pairs, n_layers, d_model]
plus the config that produced them.

Why window-mean and not the single final token (2026-08-27 fix)
------------------------------------------------------------------
The original version kept only the raw final-token activation. That token is
whatever word the free-form reasoning happened to end on ("...decision." vs.
"...now."), which differs systematically between the aligned and misaligned
sides just from author style — a lexical/token-identity shortcut, not a
"the model has reasoned about deception" signal. It showed up as a bug: even
layer 0 (one block deep) hit ~perfect held-out AUROC, which a single
transformer block cannot do by actually representing a strategic concept — it
can only do it by reading off token identity. Averaging over the last `window`
positions removes the single-token idiosyncrasy while still living entirely
inside the reasoning span (mean reasoning length ~64 words, so window=8 is
comfortably inside it for all 30 pairs) and keeps the extraction position
representationally consistent with deployment: monitor.py's CircuitBreaker
already smooths its live per-token scores with a rolling mean over the same
`window` (default 8) before comparing to tau. Train-time pooling now matches
inference-time pooling.

Usage:
    python src/extract.py --model qwen3-4b --pairs data/pairs.json --out data/acts.pt
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from data_utils import load_pairs, get_texts_for_extraction
import models as M


def windowed_resid(model, text: str, window: int = 8) -> torch.Tensor:
    """Mean residual stream over the last `window` tokens, for every layer.

    Returns [n_layers, d_model].

    names_filter is not optional at scale: caching every hook point on a 4B+
    model will exhaust a T4. We only ever read resid_post.
    """
    with torch.no_grad():
        _, cache = model.run_with_cache(
            text,
            names_filter=lambda n: n.endswith("resid_post"),
            return_type=None,
        )
    acts = torch.stack([
        cache["resid_post", L][0, -window:, :].mean(dim=0)
        for L in range(model.cfg.n_layers)
    ])
    del cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return acts.float().cpu()


def extract(model, pairs: list[dict], window: int = 8, show_progress: bool = True):
    """Extract activations for all pairs.

    Returns (aligned [n,L,d], misaligned [n,L,d], pair_ids).
    """
    aligned_texts, misaligned_texts, pair_ids = get_texts_for_extraction(pairs)

    A, Mis = [], []
    it = zip(aligned_texts, misaligned_texts)
    if show_progress:
        it = tqdm(list(it), desc="extracting")

    for a_text, m_text in it:
        A.append(windowed_resid(model, a_text, window=window))
        Mis.append(windowed_resid(model, m_text, window=window))

    return torch.stack(A), torch.stack(Mis), pair_ids


def main():
    p = argparse.ArgumentParser(description="Extract residual-stream activations")
    p.add_argument("--model", default="qwen3-4b", help="key from src/models.py REGISTRY")
    p.add_argument("--pairs", default="data/pairs.json")
    p.add_argument("--out", default="data/acts.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--window", type=int, default=8,
                    help="mean-pool the last N reasoning tokens (matches monitor.py's smoothing window)")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    pairs = load_pairs(args.pairs)
    print(f"{len(pairs)} pairs from {args.pairs}")

    entry = M.REGISTRY[args.model]
    print(f"model: {args.model} -> {entry.paper_flagship} [{entry.lineage}]")

    model = M.load(args.model, dtype=args.dtype)

    t0 = time.time()
    A, Mis, pair_ids = extract(model, pairs, window=args.window)
    elapsed = time.time() - t0

    assert A.shape == Mis.shape, f"shape mismatch: {A.shape} vs {Mis.shape}"
    assert A.shape[0] == len(pairs), "wrong number of pairs extracted"

    config = {
        "model_key": args.model,
        "hf_id": entry.hf_id,
        "paper_flagship": entry.paper_flagship,
        "lineage": entry.lineage,
        "n_layers": model.cfg.n_layers,
        "d_model": model.cfg.d_model,
        "n_pairs": len(pairs),
        "seed": args.seed,
        "dtype": args.dtype,
        "pairs_file": args.pairs,
        "position": f"last_{args.window}_mean",
        "window": args.window,
        "hook": "resid_post",
        "elapsed_s": round(elapsed, 1),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"aligned": A, "misaligned": Mis,
                "pair_ids": pair_ids, "config": config}, args.out)

    print(f"\naligned    {tuple(A.shape)}")
    print(f"misaligned {tuple(Mis.shape)}")
    print(f"[n_pairs, n_layers, d_model] in {elapsed:.0f}s -> {args.out}")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
