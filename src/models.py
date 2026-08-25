"""
models.py — Model registry for the Lynch et al. replication.

Maps the open-weight models tested in Lynch et al. 2025 (arXiv:2510.05179) to
locally-runnable HuggingFace models, with an explicit record of how strong the
lineage claim is in each case.

The three open-weight models in the paper are:
    DeepSeek-R1        671B MoE (37B active)   MIT
    Llama 4 Maverick   400B MoE (17B active)   Llama 4 Community License
    Qwen3-235B-A22B    235B MoE (22B active)   Apache 2.0

None fit on Colab. Each entry below records the flagship, the proxy actually
used, and — importantly — how defensible it is to treat the proxy as standing
in for the flagship. Do not let "we tested DeepSeek" appear in a writeup when
what was tested is a 7B distill.
"""

from dataclasses import dataclass, field


# ── Registry ──────────────────────────────────────────────────────────────────

@dataclass
class ModelEntry:
    key: str
    hf_id: str
    params_b: float
    paper_flagship: str            # the model Lynch et al. actually tested
    lineage: str                   # "same-family" | "official-distill" | "different-generation"
    lineage_note: str              # what can honestly be claimed
    arch: str                      # underlying architecture (matters for TransformerLens)
    tl_ok: bool                    # hooks cleanly in TransformerLens
    notes: str = ""

    @property
    def fp16_gb(self) -> float:
        return self.params_b * 2

    @property
    def int4_gb(self) -> float:
        return self.params_b * 0.5

    def fits(self, vram_gb: float, dtype: str = "fp16") -> bool:
        """Rough fit check. ~1.4x headroom for activations, KV cache, and the
        TransformerLens cache itself (which is not small at long context)."""
        need = self.fp16_gb if dtype == "fp16" else self.int4_gb
        return need * 1.4 <= vram_gb


REGISTRY: dict[str, ModelEntry] = {

    # ── Qwen3 lineage — the strongest claim of the three ──────────────────────
    # Qwen3-235B-A22B was tested in the paper. The small Qwen3 models are the
    # same release, same post-training recipe, same tokenizer — genuinely the
    # same family, just a different size point.
    "qwen3-8b": ModelEntry(
        key="qwen3-8b",
        hf_id="Qwen/Qwen3-8B",
        params_b=8.2,
        paper_flagship="Qwen3-235B-A22B",
        lineage="same-family",
        lineage_note=(
            "Same family and release generation as the paper's Qwen3-235B-A22B. "
            "Differs in scale and in being dense rather than MoE. Claim: 'a smaller "
            "member of the same model family' — NOT 'the model the paper tested'."
        ),
        arch="qwen3",
        tl_ok=True,
        notes="fp16 needs ~16GB — A100 yes, T4 no. Use qwen3-4b on a T4.",
    ),
    "qwen3-4b": ModelEntry(
        key="qwen3-4b",
        hf_id="Qwen/Qwen3-4B",
        params_b=4.0,
        paper_flagship="Qwen3-235B-A22B",
        lineage="same-family",
        lineage_note="Same family as Qwen3-235B-A22B, two size tiers down.",
        arch="qwen3",
        tl_ok=True,
        notes="fp16 ~8GB — comfortable on a T4. Default choice for free Colab.",
    ),

    # ── DeepSeek-R1 lineage ──────────────────────────────────────────────────
    # The distills are official DeepSeek releases, post-trained on 800k samples
    # of R1's own chain-of-thought. Architecturally they are Qwen2.5 / Llama3.1,
    # NOT MLA — which is why they hook cleanly where R1 itself would not.
    "r1-distill-qwen-7b": ModelEntry(
        key="r1-distill-qwen-7b",
        hf_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        params_b=7.6,
        paper_flagship="DeepSeek-R1",
        lineage="official-distill",
        lineage_note=(
            "Official DeepSeek release, distilled from DeepSeek-R1's own CoT onto a "
            "Qwen2.5-7B base. Inherits R1's reasoning style, not its weights. Claim: "
            "'a distillation of the model tested in the paper' — and say which base."
        ),
        arch="qwen2",
        tl_ok=True,
        notes="Base is Qwen2.5-7B, so no MLA — hooks fine. fp16 ~15GB, tight on T4.",
    ),
    "r1-distill-qwen-1.5b": ModelEntry(
        key="r1-distill-qwen-1.5b",
        hf_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        params_b=1.8,
        paper_flagship="DeepSeek-R1",
        lineage="official-distill",
        lineage_note="Same distillation, Qwen2.5-1.5B base. Smallest usable R1 descendant.",
        arch="qwen2",
        tl_ok=True,
        notes="fp16 ~3.5GB. Fastest option; weakest capability.",
    ),
    "r1-distill-llama-8b": ModelEntry(
        key="r1-distill-llama-8b",
        hf_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        params_b=8.0,
        paper_flagship="DeepSeek-R1",
        lineage="official-distill",
        lineage_note=(
            "Official R1 distill onto Llama-3.1-8B-Base. Useful as an architecture "
            "control against r1-distill-qwen-7b: same distillation, different base."
        ),
        arch="llama",
        tl_ok=True,
        notes="Pairs with r1-distill-qwen-7b to separate base-model effects from R1 CoT.",
    ),

    # ── Llama lineage — the weakest claim ────────────────────────────────────
    # Llama 4 Maverick is 400B and Scout, the small one, is still 109B. There is
    # no small Llama 4. Llama-3.1-8B is a different generation and should be
    # described that way.
    "llama-3.1-8b": ModelEntry(
        key="llama-3.1-8b",
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        params_b=8.0,
        paper_flagship="Llama 4 Maverick",
        lineage="different-generation",
        lineage_note=(
            "WEAK LINEAGE. Llama 4 has no small variant (Scout is 109B), so this is a "
            "previous-generation Llama, not a small Llama 4. Do not present it as "
            "standing in for Maverick. Include it as a Llama-family datapoint or omit it."
        ),
        arch="llama",
        tl_ok=True,
        notes="Gated on HF — accept the license and set HF_TOKEN before loading.",
    ),
}


# The flagships themselves, for the record. Not runnable — listed so the writeup
# can state exactly what was and wasn't tested.
PAPER_FLAGSHIPS = {
    "DeepSeek-R1":      dict(params_b=671, active_b=37, license="MIT",
                             hf_id="deepseek-ai/DeepSeek-R1", runnable_locally=False),
    "Llama 4 Maverick": dict(params_b=400, active_b=17, license="Llama 4 Community",
                             hf_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
                             runnable_locally=False),
    "Qwen3-235B-A22B":  dict(params_b=235, active_b=22, license="Apache-2.0",
                             hf_id="Qwen/Qwen3-235B-A22B", runnable_locally=False),
}


# ── Selection ─────────────────────────────────────────────────────────────────

def recommend(vram_gb: float) -> list[str]:
    """Best one model per paper-lineage that fits in the given VRAM.

    Returns keys in a sensible run order. Prefers larger within a lineage, since
    capability floors matter a lot for agentic scenarios — a model that can't
    follow the setup produces noise, not a null result.
    """
    by_flagship: dict[str, list[ModelEntry]] = {}
    for e in REGISTRY.values():
        if e.fits(vram_gb) and e.tl_ok:
            by_flagship.setdefault(e.paper_flagship, []).append(e)

    picked = []
    for flagship, entries in by_flagship.items():
        best = max(entries, key=lambda e: e.params_b)
        picked.append(best.key)
    # strongest lineage claim first
    order = {"same-family": 0, "official-distill": 1, "different-generation": 2}
    return sorted(picked, key=lambda k: order[REGISTRY[k].lineage])


def report(vram_gb: float = 16.0) -> None:
    """Print the full picture: what the paper tested, what fits, what can be claimed."""
    print(f"Lynch et al. (arXiv:2510.05179) — open-weight models tested\n{'='*74}")
    for name, d in PAPER_FLAGSHIPS.items():
        print(f"  {name:<18} {d['params_b']:>4}B total / {d['active_b']:>2}B active   "
              f"{d['license']:<22} local: {'yes' if d['runnable_locally'] else 'NO'}")

    print(f"\n\nLocally-runnable proxies (VRAM budget: {vram_gb:.0f}GB)\n{'='*74}")
    print(f"{'key':<22}{'params':>8}{'fp16':>8}{'4bit':>7}  {'fits':<6}{'lineage':<22}")
    print("-" * 74)
    for e in REGISTRY.values():
        fit = "✓" if e.fits(vram_gb) else ("4bit" if e.fits(vram_gb, "int4") else "✗")
        print(f"{e.key:<22}{e.params_b:>7.1f}B{e.fp16_gb:>7.0f}G{e.int4_gb:>6.0f}G  "
              f"{fit:<6}{e.lineage:<22}")

    rec = recommend(vram_gb)
    print(f"\n\nRecommended set ({len(rec)} models, one per lineage)\n{'='*74}")
    for k in rec:
        e = REGISTRY[k]
        print(f"\n  {e.key}  →  stands in for {e.paper_flagship}")
        print(f"    {e.hf_id}")
        print(f"    {e.lineage_note}")
        if e.notes:
            print(f"    note: {e.notes}")

    print(f"\n\n{'='*74}")
    print("Quantization warning: 4-bit changes activation values. Fine for behavioral")
    print("runs (Exp 0), NOT fine for reading-vector extraction (Exp 1/2) unless every")
    print("model in the comparison is quantized identically. Prefer fp16 for anything")
    print("that touches the residual stream.")


# ── Loading ───────────────────────────────────────────────────────────────────

def load(key: str, device: str = "cuda", dtype: str = "fp16"):
    """Load a registry model as a hookable TransformerLens model.

    TransformerLens 3 deprecated HookedTransformer.from_pretrained in favour of
    TransformerBridge. We try the bridge first and fall back, so this works on
    both v2 and v3 installs.
    """
    import torch
    entry = REGISTRY[key]
    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16

    try:                                            # TransformerLens >= 3
        from transformer_lens import TransformerBridge
        model = TransformerBridge.boot_transformers(
            entry.hf_id, device=device, dtype=torch_dtype)
        api = "TransformerBridge"
    except ImportError:                             # TransformerLens 2.x
        import warnings
        from transformer_lens import HookedTransformer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            model = HookedTransformer.from_pretrained(
                entry.hf_id, device=device, dtype=torch_dtype)
        api = "HookedTransformer (deprecated)"

    model.eval()
    print(f"loaded {entry.key} via {api} | {model.cfg.n_layers} layers | "
          f"d_model={model.cfg.d_model}")
    return model


if __name__ == "__main__":
    import sys
    vram = float(sys.argv[1]) if len(sys.argv) > 1 else 16.0
    report(vram)