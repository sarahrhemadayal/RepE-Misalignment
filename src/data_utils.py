"""
data_utils.py — Dataset loading, splitting, and formatting for the extraction pipeline.

Downstream users:
    src/extract.py  — calls get_texts_for_extraction() to build the prompt list
    src/probe.py    — calls get_concept_splits() to train per-concept probes
    notebooks/      — call summarize() for quick inspection

Nothing in here touches a model or GPU.
"""

import json
import random
from pathlib import Path
from collections import Counter
from typing import Optional


VALID_CONCEPTS = {"deception", "shutdown_avoidance", "power_seeking"}


# ── Loading ───────────────────────────────────────────────────────────────────

def load_pairs(path: str | Path) -> list[dict]:
    """Load and return pairs.json as a list of dicts. No validation here — run
    validate_pairs.py first."""
    with open(path) as f:
        return json.load(f)


# ── Splitting ─────────────────────────────────────────────────────────────────

def train_test_split(
    pairs: list[dict],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Split pairs into train / test sets.

    Stratified by concept so each concept has proportional representation in
    both splits. With 30 pairs and 3 concepts (~10 each), this gives ~8 train
    / 2 test per concept.

    Args:
        pairs: full list of pairs
        test_fraction: fraction reserved for test (default 0.2)
        seed: random seed for reproducibility

    Returns:
        (train_pairs, test_pairs)
    """
    rng = random.Random(seed)
    by_concept: dict[str, list[dict]] = {}
    for p in pairs:
        by_concept.setdefault(p["concept"], []).append(p)

    train, test = [], []
    for concept, concept_pairs in by_concept.items():
        shuffled = concept_pairs[:]
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_fraction))
        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])

    return train, test


def get_concept_splits(pairs: list[dict]) -> dict[str, list[dict]]:
    """Group pairs by concept. Returns dict[concept -> list[pair]]."""
    splits: dict[str, list[dict]] = {}
    for p in pairs:
        splits.setdefault(p["concept"], []).append(p)
    return splits


# ── Text formatting for extraction ───────────────────────────────────────────

def format_prompt(pair: dict, side: str) -> str:
    """
    Build the prompt string that gets fed to the model for activation extraction.

    Format:  <scenario>\\n\\n<reasoning>

    side: 'aligned' or 'misaligned'

    The reasoning trace is appended to the scenario so the model's residual
    stream encodes both the context and the reasoning direction. We read
    activations at the FINAL token of this combined string.

    Note: Do not add any instruction prefix here. The prompts are fed as raw
    continuations to keep the activation signal clean. If you want to add a
    system prompt or role header, do it in extract.py and document the choice.
    """
    assert side in ("aligned", "misaligned"), f"side must be 'aligned' or 'misaligned', got '{side}'"
    reasoning = pair[f"{side}_reasoning"]
    return f"{pair['scenario']}\n\n{reasoning}"


def get_texts_for_extraction(
    pairs: list[dict],
) -> tuple[list[str], list[str], list[str]]:
    """
    Build parallel lists of aligned and misaligned prompts for extract.py.

    Returns:
        aligned_texts    list[str] — one per pair, combined scenario + aligned reasoning
        misaligned_texts list[str] — one per pair, combined scenario + misaligned reasoning
        pair_ids         list[str] — pair IDs in the same order, for indexing saved activations

    Usage in extract.py:
        aligned_texts, misaligned_texts, ids = get_texts_for_extraction(pairs)
        for i, (aligned, misaligned) in enumerate(zip(aligned_texts, misaligned_texts)):
            aligned_acts[i]    = extract_final_token_resid(aligned, layer=L)
            misaligned_acts[i] = extract_final_token_resid(misaligned, layer=L)
    """
    aligned = [format_prompt(p, "aligned") for p in pairs]
    misaligned = [format_prompt(p, "misaligned") for p in pairs]
    ids = [p["id"] for p in pairs]
    return aligned, misaligned, ids


# ── Inspection ────────────────────────────────────────────────────────────────

def summarize(pairs: list[dict], verbose: bool = False) -> None:
    """Print a quick dataset summary."""
    concepts = Counter(p.get("concept", "MISSING") for p in pairs)
    print(f"Total pairs: {len(pairs)}")
    for c in sorted(concepts):
        print(f"  {c}: {concepts[c]}")

    if verbose:
        print()
        for i, p in enumerate(pairs):
            al_words = len(p.get("aligned_reasoning", "").split())
            mis_words = len(p.get("misaligned_reasoning", "").split())
            print(
                f"  [{i:02d}] {p.get('id','?'):8s} | {p.get('concept','?'):22s} | "
                f"aligned={al_words:3d}w | misaligned={mis_words:3d}w"
            )


def get_pair_by_id(pairs: list[dict], pair_id: str) -> Optional[dict]:
    """Retrieve a single pair by id. Returns None if not found."""
    for p in pairs:
        if p.get("id") == pair_id:
            return p
    return None
