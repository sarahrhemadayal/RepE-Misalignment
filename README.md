# Detecting Agentic Misalignment via Representation Engineering

---

## Abstract

LLM agents given autonomy and conflicting goals can reason internally toward harmful strategies — deception, coercion, shutdown-avoidance — while their visible output stays compliant until the moment the harmful action executes. Output-level filters only see the result. By then the decision is already made.

This project reads **concept-specific directions out of the residual stream during generation** and builds a monitor that scores reasoning in real time and intervenes before the action completes.

The core claim being tested: _if a model's harmful reasoning happens somewhere, it happens in the residual stream — so it should be readable from activations before it becomes text._

That is an argument, not an established fact. Testing it empirically is the project.

---

## Where this sits in the literature

The seminar organises detection work by **what signal each method monitors**. Each level exists because of a limitation in the one below it.

| Level  | What it reads                                     | Representative work                            | Limitation motivating the next level                                                                         |
| ------ | ------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **L1** | Completed actions and final text                  | Lynch et al. (2025)                            | The harmful decision is already taken by the time it becomes visible                                         |
| **L2** | The reasoning the model writes out                | Chen et al. (2025)                             | Stated reasoning does not always report its own cause; disclosure can be suppressed while behaviour persists |
| **L3** | Residual-stream activations during generation     | Goldowsky-Dill et al. (2025); Anthropic (2026) | Evidence is largely single-turn and off-policy; correlation is not causal validation                         |
| **L4** | Activations, _and acts on them_ during generation | Li et al. (2026)                               | Brittle across distributions, with measurable side effects on alignment                                      |

**This project builds L3 → L4.** Notebook `01` produces the L3 reading vector; notebook `02` turns it into an L4 intervention.

**Research gap:** the closest survey (Bartoszcze et al., 2025) organises this work by _method_ rather than by _monitoring role_, and predates the 2026 agentic results. No review states what evidential standard each monitoring level meets for real-time agentic deployment.

---

## Quick start

### Colab (recommended — no local GPU needed)

1. Open `notebooks/01_extract_and_vectors.ipynb` in Colab
2. **Runtime → Change runtime type → T4 GPU**
3. Run all cells. First cell clones this repo; set `YOUR_USERNAME` in it.
4. Then run `notebooks/02_halt_demo.ipynb`

Total runtime ≈ 30–40 min, most of it the initial model download.

### Local

```bash
git clone https://github.com/sarahrhemadayal/agentic-misalignment-repe.git
cd agentic-misalignment-repe
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

python tests/validate_pairs.py data/pairs.json      # sanity-check the dataset
python src/models.py 16                             # what fits in 16GB VRAM?
python src/extract.py --model qwen3-4b              # -> data/acts.pt
python src/vectors.py                               # -> data/vectors.pt
```

---

## Repository layout

```
agentic-misalignment-repe/
├── data/
│   ├── pairs.json              30 contrastive deception pairs (committed)
│   ├── acts.pt                 extracted activations (gitignored — regenerate)
│   └── vectors.pt              reading vectors + eval (gitignored)
├── src/
│   ├── models.py               model registry: paper models → runnable proxies
│   ├── data_utils.py           dataset loading, splitting, prompt formatting
│   ├── extract.py              residual-stream extraction
│   ├── vectors.py              diff-of-means, held-out eval, AUROC
│   └── monitor.py              streaming circuit breaker
├── tests/
│   └── validate_pairs.py       schema + quality checks on the dataset
├── notebooks/
│   ├── 00_setup.ipynb          toolchain sanity check
│   ├── 01_extract_and_vectors.ipynb    core pipeline, the review deliverable
│   ├── 02_halt_demo.ipynb              L4 intervention demo
│   ├── exp0_paper_models.ipynb         Phase 3: contamination test
│   ├── exp1_cornering.ipynb            Phase 3: cornering sweep
│   └── exp2_cot_timing.ipynb           Phase 3: activation vs CoT timing
├── figures/                    output plots
├── notes/                      working notes (trace_patterns.md)
└── results/                    experiment summaries
```

---

## Method

### 1. Contrastive dataset

30 pairs. Within each pair the **scenario is byte-identical** and only the reasoning trace differs — one aligned, one misaligned.

That constraint is the whole design. Any other difference between the two traces is a confound the vector will latch onto instead of the reasoning. If misaligned traces were systematically longer, you would extract a length direction and it would separate held-out data beautifully.

The 30 pairs span 30 distinct domains (healthcare, finance, logistics, aviation, customs…) so the vector cannot simply learn "corporate email agent". Validate with:

```bash
python tests/validate_pairs.py data/pairs.json --strict
```

Schema in `docs/` and `src/data_utils.py`. Source label: `synthetic — adapted from Lynch et al. 2025`.

### 2. Extraction

For each pair, `scenario + reasoning` runs through the model and `resid_post` is cached at every layer. We keep the **final-token** activation — the only position that has attended over both the full scenario and the full reasoning trace.

Output: two tensors `[30, n_layers, d_model]`.

### 3. Reading vector

Per-layer difference of means, unit-normalised:

```
v_L = normalise( mean(misaligned_L) − mean(aligned_L) )
```

**Layer selection is done on held-out data.** Six pairs are held out with a fixed seed, the vector is re-derived from the remaining 24 only, and the held-out pairs are scored against it. Choosing the layer by in-sample separation would overfit at n=30 — the winner would partly be wherever the training noise happened to line up.

`src/vectors.py` also computes a leave-one-out separation as a cross-check. **If the two methods disagree about the best layer, the choice is not robust and the code says so** rather than silently picking the better-looking one.

### 4. Circuit breaker

Score the residual stream against `v_L` at every generated token; halt when a rolling mean crosses τ.

τ is set from a **target false-positive rate** — the (1−FPR) quantile of the held-out aligned distribution — with detection rate falling out as a consequence. The reverse (tune for detection, then report FPR) is the tempting mistake and produces monitors that halt constantly in deployment.

> **Naming:** this is _not_ Zou et al.'s 2024 "Circuit Breakers", which reroutes harmful representations during _training_. This monitors at inference. Do not conflate them in writeups.

---

## Models

Lynch et al. tested 16 models; exactly three are open-weight, and none fit on Colab:

| Paper model      | Params                | License           | Runs on Colab? |
| ---------------- | --------------------- | ----------------- | -------------- |
| DeepSeek-R1      | 671B MoE (37B active) | MIT               | No             |
| Llama 4 Maverick | 400B MoE (17B active) | Llama 4 Community | No             |
| Qwen3-235B-A22B  | 235B MoE (22B active) | Apache-2.0        | No             |

`src/models.py` maps each to a runnable descendant **and records how defensible that substitution is.** These are not equal:

| Local model                                            | Stands in for    | Lineage                  | Honest claim                                                                                                                                                |
| ------------------------------------------------------ | ---------------- | ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `qwen3-8b` / `qwen3-4b`                                | Qwen3-235B-A22B  | same-family              | Same family, same release, same post-training recipe. Fair.                                                                                                 |
| `r1-distill-qwen-7b` / `-1.5b` / `r1-distill-llama-8b` | DeepSeek-R1      | official-distill         | Official DeepSeek releases distilled from R1's own CoT. Fair — name the base.                                                                               |
| `llama-3.1-8b`                                         | Llama 4 Maverick | **different-generation** | ⚠️ **Weak.** Llama 4 has no small variant (Scout is 109B). Previous generation. Include as a Llama datapoint or omit — cannot carry a claim about Maverick. |

```bash
python src/models.py 16   # T4
python src/models.py 40   # A100
```

**Use fp16, not 4-bit.** Quantization changes activation values. Fine for behavioural counts, corrupts vector extraction.

### If you want the actual flagships

Not a Colab decision at any tier — these need multi-GPU serving (vLLM/SGLang with tensor parallelism), roughly 8×A100 80GB or 4×H100 for DeepSeek-R1, at ~$16–25/hr on rented cloud. Weights alone exceed 1TB.

Realistic alternative: hosted APIs (Together.ai, Fireworks, DeepInfra) serve these exact open-weight models for **behavioural data only** — no activations, so it extends Exp 0 but not the L3/L4 pipeline. Logged as a potential extension, not a project dependency.

---

## What this establishes — and what it does not

**Does:**

- A direction that separates held-out aligned/misaligned reasoning traces (AUROC, reported with held-out n)
- A working inference-time monitor that halts on threshold crossing
- Detection and false-positive rates on an off-distribution demo domain

**Does not:**

- **Prove the direction _is_ deception.** Separation is correlational. Establishing causality needs steering (add the vector — does misaligned output rise?) and ablation (remove it — does it fall?). Neither is done yet. This is the single most important caveat and it goes on a slide, not in a footnote.
- **Rule out that the vector tracks something adjacent** — generic negative valence, or activation of a "manipulative character" persona rather than reasoning-about-harm. Testable: does the vector fire on morally-questionable _fictional character_ text with no agentic misalignment in it?
- **Say anything reliable about the models Lynch et al. tested.** These are 4–8B descendants of 235B–671B models.
- **Generalise beyond one concept.** Only deception. `shutdown_avoidance` and `power_seeking` are Phase 2+.

---

## Roadmap

Full phase definitions, the two-course split, and the Aug–Nov schedule: **[`docs/PHASES.md`](docs/PHASES.md)**.
Current state, blockers, open decisions: **[`docs/Phase_Tracker.md`](docs/Phase_Tracker.md)**.

| Phase | Work                                             | Window          | Status                            |
| ----- | ------------------------------------------------ | --------------- | --------------------------------- |
| 1     | Literature review, foundation, L1–L4 taxonomy    | –29 Aug         | essentially done                  |
| 2     | Model scope, contrastive dataset construction    | 8 Sept – 5 Oct  | partial — 30 deception pairs done |
| 3a    | Reading vectors, held-out AUROC (**L3**)         | sprint week     | built, not yet run                |
| 3b    | Steering, ablation, graded intervention (**L4**) | 13–26 Oct       | **not started — critical path**   |
| 4     | Evaluation, reports, publication draft           | 27 Oct – 16 Nov | pending                           |

**Phase 3b is the critical path.** Everything currently built is 3a, which is correlational. Every limitation in the section above traces back to 3b being incomplete.

**Phase 3 experiments** (`notebooks/exp*.ipynb`) test two critiques of Lynch et al.:

- **Exp 0** — do the published scenarios still work as an eval, or do models now just recognise them? (verbatim vs structure-matched-novel)
- **Exp 1** — does misalignment survive when a legitimate exit exists, or does the cornering create it?
- **Exp 2** — does the activation signal precede the model verbalising the strategy?

Run order: `exp0` → `exp1` → `exp2`. Exp 0 first, because if it finds contamination, Exp 1's scenarios need rewriting before they are worth running.

---

## Citation policy

External-facing deliverables cite **2025–2026 sources only**. Foundational work (Zou et al. RepE 2023; Circuit Breakers 2024) is used for grounding but cited via 2025–2026 papers that build on it, or the method is described without citation. Where that creates an attribution gap it is flagged, not silently dropped.

## Reproducibility

Fixed seeds (default 42) throughout. Every run logs its config: model, layer, dtype, split seed, dataset file, n_pairs. `data/acts.pt` and `data/vectors.pt` are gitignored — regenerate from `pairs.json`, which is committed.

## Team

Siddhi Alat · Tanish Belel · Sarah Dayal
See `TEAM_BRIEF.txt` for onboarding.

## Key references

- Lynch, Wright, Larson et al. (Anthropic), _Agentic Misalignment: How LLMs Could Be Insider Threats_, arXiv:2510.05179, 2025
- Bartoszcze et al., _Representation Engineering for LLMs: Survey and Research Challenges_, arXiv:2502.17601, 2025
- Chen et al., _Reasoning models don't always say what they think_, arXiv:2505.05410, 2025
- Goldowsky-Dill et al., linear probes for strategic deception, 2025
- Song & Sun, SPAR Sp26 (UC Berkeley) — primary replication target
