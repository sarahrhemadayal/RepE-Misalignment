"""
monitor.py — Streaming circuit breaker.

Scores the residual stream against a reading vector at every generated token and
halts generation when a running score crosses a threshold.

This is the "circuit breaker" of the project title, and it is NOT Zou et al.'s
2024 Circuit Breakers — that method reroutes harmful representations during
training. This monitors at inference and intervenes on the generation loop.

Design note on smoothing: raw per-token projections are noisy enough that a bare
threshold fires on single-token spikes. We halt on a rolling mean over
`window` tokens instead, which trades a few tokens of detection latency for a
large drop in false halts. The latency cost is reported by the demo so it is
visible rather than hidden.

Usage (as a library):
    from monitor import CircuitBreaker
    cb = CircuitBreaker(model, vector, layer=14, tau=0.8)
    result = cb.generate(prompt, max_new_tokens=300)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class MonitorResult:
    text: str
    scores: list[float] = field(default_factory=list)   # smoothed, per token
    raw_scores: list[float] = field(default_factory=list)
    halted: bool = False
    halt_index: int | None = None
    n_tokens: int = 0

    def summary(self) -> str:
        if self.halted:
            return (f"[HALTED at token {self.halt_index}/{self.n_tokens}] "
                    f"peak={max(self.scores):.3f}")
        peak = max(self.scores) if self.scores else float("nan")
        return f"[completed {self.n_tokens} tokens] peak={peak:.3f}"


class CircuitBreaker:
    """Token-by-token monitor with a halt threshold."""

    def __init__(self, model, vector: torch.Tensor, layer: int,
                 tau: float = 1.0, window: int = 8):
        self.model = model
        self.vector = vector.float()
        self.layer = layer
        self.tau = tau
        self.window = window
        self.hook_name = f"blocks.{layer}.hook_resid_post"

    # ── scoring ──────────────────────────────────────────────────────────────

    def score_text(self, text: str, skip: int = 0) -> np.ndarray:
        """Per-token projection for a complete string.

        `skip` drops the first N positions (use it to exclude the prompt).

        Because the model is causal, running the finished text through in one
        pass reproduces exactly the activations that existed during generation —
        position t attends only to < t. This is not an approximation, and it is
        far cheaper than hooking every generation step.
        """
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                text, names_filter=lambda n: n == self.hook_name, return_type=None)
        resid = cache["resid_post", self.layer][0].float().cpu()
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (resid @ self.vector).numpy()[skip:]

    @staticmethod
    def smooth(scores, window: int) -> list[float]:
        """Trailing rolling mean."""
        buf, out = deque(maxlen=window), []
        for s in scores:
            buf.append(float(s))
            out.append(sum(buf) / len(buf))
        return out

    # ── generation ───────────────────────────────────────────────────────────

    def generate(self, prompt: str, max_new_tokens: int = 300,
                 temperature: float = 0.8, seed: int | None = None,
                 verbose: bool = False) -> MonitorResult:
        """Generate, then score and apply the halt rule.

        Note this generates first and evaluates after, rather than genuinely
        interrupting the sampling loop. For a causal model the scores are
        identical either way, so the halt decision is the same one a true
        inline monitor would have made at that token — but the text after the
        halt point was still computed. That distinction matters if you ever
        claim a compute saving; it does not affect the detection claim.
        Truncating the returned text is what makes it behave as a halt.
        """
        if seed is not None:
            torch.manual_seed(seed)

        n_prompt = len(self.model.to_tokens(prompt)[0])

        with torch.no_grad():
            out = self.model.generate(prompt, max_new_tokens=max_new_tokens,
                                      temperature=temperature, do_sample=True,
                                      verbose=False)
        full = out if isinstance(out, str) else self.model.to_string(out[0])

        raw = self.score_text(full, skip=n_prompt)
        sm = self.smooth(raw, self.window)

        # Don't evaluate the halt rule until the rolling window is actually full --
        # before that, `sm[i]` is an average over fewer than `window` raw scores (sm[0]
        # is literally one unsmoothed token), exactly the single-token-spike noise the
        # smoothing exists to filter out. Skipping the first `window - 1` positions was
        # missing before and caused spurious near-instant halts on both aligned and
        # misaligned runs alike (2026-08-27 fix).
        halt_idx = next((i for i, s in enumerate(sm) if i >= self.window - 1 and s > self.tau), None)

        toks = self.model.to_str_tokens(full)[n_prompt:]
        if halt_idx is not None:
            text = "".join(toks[:halt_idx + 1])
        else:
            text = full[len(prompt):]

        res = MonitorResult(text=text, scores=sm, raw_scores=list(map(float, raw)),
                            halted=halt_idx is not None, halt_index=halt_idx,
                            n_tokens=len(sm))

        if verbose:
            print(res.summary())
        return res


# ── Threshold selection ───────────────────────────────────────────────────────

def choose_tau(aligned_scores: np.ndarray, misaligned_scores: np.ndarray,
               target_fpr: float = 0.05) -> dict:
    """Pick tau from held-out scores at a target false-positive rate.

    Setting tau by eye on a histogram is how you end up with a threshold that
    only works on the data you looked at. This sets it to the (1 - target_fpr)
    quantile of the ALIGNED distribution, so the false-positive rate is chosen
    deliberately and the detection rate is whatever falls out — rather than the
    other way round, which is the tempting mistake.
    """
    tau = float(np.quantile(aligned_scores, 1 - target_fpr))
    fpr = float((aligned_scores > tau).mean())
    tpr = float((misaligned_scores > tau).mean())
    return {"tau": tau, "fpr": fpr, "tpr": tpr, "target_fpr": target_fpr}
