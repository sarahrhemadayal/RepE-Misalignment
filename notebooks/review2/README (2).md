# Gemma 3 12B act/refrain interpretability: one notebook per phase

Each notebook runs on its own Google Colab runtime. Runtimes do not share a disk, so files move between them through **one Google Drive folder** (`MyDrive/gemma-interp`), and the small input files are fetched from your **GitHub repo**.

## How files are found
Every notebook calls `need("<file>")`, which looks in this order:
1. `MyDrive/gemma-interp/<file>` (mounted at the top of each notebook)
2. `REPO_RAW + <file>`: set `REPO_RAW` in the first cell to your public raw GitHub folder, e.g. `https://raw.githubusercontent.com/<you>/<repo>/main/gemma-interp/`
3. otherwise it stops and tells you exactly which file is missing and which notebook makes it.

Outputs are written to `MyDrive/gemma-interp/` and `MyDrive/gemma-interp/results/`, so they survive a disconnect. Generation and grading save after every item (Drive writes are slow but safe) and resume if you rerun.

## Files to upload to GitHub
| File | Needed by | Source | Size |
|---|---|---|---|
| the 7 `.ipynb` files | - | this folder | small |
| `contrast_transcripts.json` | phases 0, 1, 3, 4, 5 | already in your repo (183 hand-checked transcripts) | 2.9 MB |
| `commit_diff_means.pt` | phases 3, 4 (5 optional) | already in your repo; phase 2 recreates it | 0.7 MB |
| `results/phase3sweep_completions.json` | phase 3, optional | already in your repo; if found, phase 3 skips generation | 0.7 MB |
| `results/phase4_completions.json` | phase 4, optional | already in your repo; same | 0.8 MB |

Phase 5 has no archived samples, so it always generates.

## Created by the notebooks
| File | Made by | Used by | Commit? |
|---|---|---|---|
| `contrast_acts.pt` (~130 MB), `contrast_acts_commit.pt` (~120 MB) | phase 1 | phase 2 | **No**, over GitHub's 100 MB limit. Keep on Drive. |
| `commit_diff_means.pt` | phase 2 | phases 3, 4 | Yes |
| `desperation_dir.pt` | phase 5 | - | Yes |
| `results/*_completions.json`, `*_graded_gemini.json`, `*.png` | phases 3-5 | - | Yes |
| `runs/*.eval` | phase -1 | - | Optional |

`.gitignore`:
```
contrast_acts.pt
contrast_acts_commit.pt
runs/
vllm.log
```

## Not needed
`contrast_spec.json`, `act_keep.json`, `act_labels.txt`, `refrain_set.json`, `verify_run.py` (now inside phase -1), `RUNBOOK.md`, the old `phase*.py` scripts and notebooks, and the old GPT-4o grade files (`phase3sweep_graded.json`, `phase4_graded.json`, `phase4_coherence.json`), which are kept only for comparison.

## Keys
No OpenAI key is used anywhere. A notebook looks for a key in the environment, then Colab Secrets, then asks you to type it (hidden, never saved in the notebook).
- `GEMINI_API_KEY`: grading in phases 3-5 and the grader in phase -1.
- `HF_TOKEN`: first download of Gemma (gated; accept the licence on its Hugging Face page first).

Optional: `GRADER_MODEL` (default `gemini-3.5-flash`; change it if your key says the model is not found), `GRADER_DELAY` (seconds between grader calls, default 1.0; raise on a free-tier key).
`LOCAL_API_KEY="dummy"` in phase -1 is a placeholder for the local vLLM server, not a real key. The `openai` package is installed there only because `inspect_ai` talks to the local server through it.

## Hardware (Colab runtime)
- Phases 0, 1, 3, 4, 5 and -1: **A100 or L40-class GPU** (Gemma 3 12B in bf16 needs about 24 GB, so a free T4 will run out of memory).
- Phase 2: CPU runtime is enough.

## Install
Phases 0, 1, 3, 4, 5 start with an install cell (keeps the installed torch, adds `transformer_lens`, `transformers`, `accelerate`, `scikit-learn`, `matplotlib`, `google-genai`). Restart the runtime if pip says it changed torch or transformers. Phase -1 installs `vllm`, `inspect-ai`, `inspect-evals[agentic_misalignment]` and `openai>=3.1.0`.

## Order and what feeds what
```
phase -1  runs/*.eval  --(hand-check, already done)-->  contrast_transcripts.json
phase 0   tooling check (no output)
phase 1   contrast_transcripts.json  ->  contrast_acts.pt, contrast_acts_commit.pt   (Drive)
phase 2   contrast_acts*.pt          ->  commit_diff_means.pt, probe plots           (Drive)
phase 3   commit_diff_means.pt       ->  results/phase3sweep_*
phase 4   commit_diff_means.pt       ->  results/phase4_*
phase 5   (own direction)            ->  desperation_dir.pt, results/phase5_*
```
You can skip -1 to 2 and run 3-5 directly with the two shipped files on GitHub.

## Grading note
The archived numbers came from GPT-4o. These notebooks grade with Gemini and save to `*_graded_gemini.json`, so rates will differ slightly. Compare with that in mind.

## Status
Syntax-checked and the file-lookup logic tested locally. **Not yet run end to end on a Colab GPU.** Phase -1 in particular is the most fragile (vLLM plus inspect_ai in Colab); it is optional because the shipped `contrast_transcripts.json` replaces it.
