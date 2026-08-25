#!/usr/bin/env python3
"""
validate_pairs.py — pairs.json schema and quality check.

Usage:
    python tests/validate_pairs.py data/pairs.json
    python tests/validate_pairs.py data/pairs.json --strict   # treat warnings as errors

Checks:
    SCHEMA    required fields, valid concept label, non-empty strings
    CONTRAST  aligned != misaligned; reasoning is substantive (>= MIN_WORDS words)
    DATASET   no duplicate IDs, no duplicate scenarios, concept distribution
    QUALITY   source label format, notes present

Does NOT check:
    Whether confounds exist (semantic — needs human audit or Claude API call in the auditor)
    Whether the reasoning is actually representative of misalignment (human judgment)
"""

import json
import sys
import argparse
from pathlib import Path
from collections import Counter


# ── Constants ─────────────────────────────────────────────────────────────────

VALID_CONCEPTS = {"deception", "shutdown_avoidance", "power_seeking"}
REQUIRED_FIELDS = {
    "id", "concept", "scenario",
    "aligned_reasoning", "misaligned_reasoning",
    "source", "notes"
}
MIN_REASONING_WORDS = 30   # soft floor; raise if reasoning is too thin to probe
MIN_PAIRS_PER_CONCEPT = 5  # recommendation; fewer → weak direction estimate
TARGET_TOTAL = 30


# ── Validators ────────────────────────────────────────────────────────────────

def check_schema(pair: dict) -> tuple[list[str], list[str]]:
    """Returns (errors, warnings) for a single pair."""
    errors, warnings = [], []
    pid = pair.get("id", "<no id>")

    # Required fields
    missing = REQUIRED_FIELDS - set(pair.keys())
    if missing:
        errors.append(f"[{pid}] Missing required fields: {sorted(missing)}")
        return errors, warnings  # can't check further without fields

    # Concept
    if pair["concept"] not in VALID_CONCEPTS:
        errors.append(
            f"[{pid}] Invalid concept '{pair['concept']}'. "
            f"Must be one of: {sorted(VALID_CONCEPTS)}"
        )

    # Non-empty strings
    for field in ["scenario", "aligned_reasoning", "misaligned_reasoning"]:
        if not pair[field].strip():
            errors.append(f"[{pid}] '{field}' is empty.")

    # Contrast exists
    if pair["aligned_reasoning"].strip() == pair["misaligned_reasoning"].strip():
        errors.append(f"[{pid}] aligned_reasoning == misaligned_reasoning. No contrast.")

    # Reasoning length
    for field in ["aligned_reasoning", "misaligned_reasoning"]:
        n = len(pair[field].split())
        if n < MIN_REASONING_WORDS:
            warnings.append(
                f"[{pid}] '{field}' is short ({n} words < {MIN_REASONING_WORDS}). "
                f"Thin reasoning → weak activation signal."
            )

    # Source label
    if not pair["source"].lower().startswith("synthetic"):
        warnings.append(
            f"[{pid}] source doesn't start with 'synthetic': '{pair['source']}'. "
            f"Expected format: 'synthetic — adapted from ...'."
        )

    # Notes
    if not pair.get("notes", "").strip():
        warnings.append(
            f"[{pid}] notes is empty. Add a rationale for why this pair is a clean contrast."
        )

    return errors, warnings


def check_dataset(pairs: list[dict]) -> tuple[list[str], list[str]]:
    """Dataset-level checks: duplicates, distribution."""
    errors, warnings = [], []

    # Duplicate IDs
    ids = [p.get("id") for p in pairs]
    for dup_id, count in Counter(ids).items():
        if count > 1:
            errors.append(f"Duplicate id '{dup_id}' appears {count} times.")

    # Duplicate scenarios (same context → measuring same instance twice)
    scenarios = [p.get("scenario", "").strip() for p in pairs]
    for dup_sc, count in Counter(scenarios).items():
        if count > 1 and dup_sc:
            errors.append(
                f"Duplicate scenario text appears {count} times "
                f"(first 80 chars: '{dup_sc[:80]}…'). "
                f"Each pair must have a unique scenario."
            )

    # Concept distribution
    concept_counts = Counter(p.get("concept") for p in pairs)
    for concept in VALID_CONCEPTS:
        n = concept_counts.get(concept, 0)
        if n == 0:
            warnings.append(
                f"Concept '{concept}' has 0 pairs. "
                f"A reading vector needs at least {MIN_PAIRS_PER_CONCEPT} pairs per concept."
            )
        elif n < MIN_PAIRS_PER_CONCEPT:
            warnings.append(
                f"Concept '{concept}' has only {n} pairs "
                f"(recommended >= {MIN_PAIRS_PER_CONCEPT})."
            )

    # Total
    if len(pairs) < TARGET_TOTAL:
        warnings.append(
            f"Only {len(pairs)} pairs (target: {TARGET_TOTAL}). "
            f"Run Block B to reach target."
        )

    return errors, warnings


# ── Main ──────────────────────────────────────────────────────────────────────

def run(path: str, strict: bool = False) -> int:
    """Returns 0 on pass, 1 on failure."""
    p = Path(path)
    if not p.exists():
        print(f"✗ File not found: {path}")
        return 1

    # Parse JSON
    try:
        with open(p) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON at {path}: {e}")
        return 1

    if not isinstance(data, list):
        print("✗ pairs.json must be a JSON array at the top level.")
        return 1

    print(f"\nValidating {path} ({len(data)} pairs)\n")

    all_errors: list[str] = []
    all_warnings: list[str] = []

    # Per-pair checks
    for pair in data:
        errs, warns = check_schema(pair)
        all_errors.extend(errs)
        all_warnings.extend(warns)

    # Dataset-level checks
    errs, warns = check_dataset(data)
    all_errors.extend(errs)
    all_warnings.extend(warns)

    # Concept distribution summary
    concept_counts = Counter(p.get("concept", "MISSING") for p in data)
    print("  Concept distribution:")
    for concept in sorted(VALID_CONCEPTS):
        bar = "█" * concept_counts.get(concept, 0)
        print(f"    {concept:<22} {concept_counts.get(concept, 0):3d}  {bar}")
    print()

    # Print results
    if all_errors:
        print(f"  Errors ({len(all_errors)}):")
        for e in all_errors:
            print(f"    ✗ {e}")
        print()

    if all_warnings:
        print(f"  Warnings ({len(all_warnings)}):")
        for w in all_warnings:
            print(f"    ⚠  {w}")
        print()

    # Final verdict
    fail = bool(all_errors) or (strict and bool(all_warnings))

    if fail:
        print(
            f"✗ Validation failed. "
            f"{len(all_errors)} error(s), {len(all_warnings)} warning(s)."
        )
        if strict and not all_errors:
            print("  (Failing because --strict is set and warnings exist.)")
        return 1
    else:
        msg = "No errors"
        if all_warnings:
            msg += f", {len(all_warnings)} warning(s) to review"
        print(f"✓ {len(data)} pairs valid. {msg}.")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate pairs.json")
    parser.add_argument("path", help="Path to pairs.json")
    parser.add_argument(
        "--strict", action="store_true",
        help="Treat warnings as errors (use before committing 30-pair set)"
    )
    args = parser.parse_args()
    sys.exit(run(args.path, strict=args.strict))
