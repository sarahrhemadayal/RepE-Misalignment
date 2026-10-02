# Gemma 3 12B act/refrain interpretability: phase notebooks

One notebook per phase. Run them in order, from a folder that holds the files below.
The notebooks read the folder they are run from (or `GEMMA_BASE` if you set it) and write into `results/`.

## Files to put in the GitHub folder

| File | Needed by | Where it comes from | Size |
|---|---|---|---|
| `phase_-1_harvest.ipynb` ... `phase_5_emotion.ipynb` (7 notebooks) | - | this folder | small |
| `contrast_transcripts.json` | phases 0, 1, 3, 4, 5 | already in your repo (183 hand-checked transcripts) | 2.9 MB |
| `commit_diff_means.pt` | phases 3, 4 (5 optional) | already in your repo; phase 2 recreates it | 0.7 MB |
| `results/phase3sweep_completions.json` | phase 3, optional | already in your repo; if present, phase 3 does not regenerate | 0.7 MB |
| `results/phase4_completions.json` | phase 4, optional | already in your repo; same idea | 0.8 MB |

Phase 5 has no archived samples in the repo, so it always generates.

## Created by the notebooks (do not commit the big ones)

| File | Made by | Size | Commit it? |
|---|---|---|---|
| `contrast_acts.pt` (notice anchor) | phase 1 | about 130 MB | **No**, over GitHub's 100 MB limit |
| `contrast_acts_commit.pt` (commit anchor) | phase 1 | about 120 MB | **No**, same reason |
| `commit_diff_means.pt` | phase 2 | 0.7 MB | Yes |
| `desperation_dir.pt` | phase 5 | small | Yes |
| `results/*_completions.json`, `results/*_graded_gemini.json`, `results/*.png` | phases 3-5 | small | Yes |
| `runs/*.eval` | phase -1 | varies | Optional |

Suggested `.gitignore`:
```
contrast_acts.pt
contrast_acts_commit.pt
runs/
vllm.log
```

## Not needed any more
`contrast_spec.json`, `act_keep.json`, `act_labels.txt`, `refrain_set.json` (labelling provenance only), `verify_run.py` (now inside phase -1), `RUNBOOK.md`, and the old `phase*.py` scripts and notebooks. Keep them if you want the history.
The old GPT-4o grade files (`phase3sweep_graded.json`, `phase4_graded.json`, `phase4_coherence.json`) are not read by the notebooks; they are kept only for comparison.

## Keys (never put these in a notebook or in GitHub)
Set them as environment variables (Colab: Secrets, then `os.environ[...] = userdata.get(...)` in a cell):
- `GEMINI_API_KEY`: grading in phases 3-5, and the grader in phase -1.
- `HF_TOKEN`: first download of Gemma (a gated model; accept the licence on its Hugging Face page first).

Optional: `GRADER_MODEL` (default `gemini-3.5-flash`; the Gemini docs list several ids, so change it if your key says the model is not found), `GRADER_DELAY` (seconds between grader calls, default 1.0; raise it on a free-tier key).
`LOCAL_API_KEY="dummy"` in phase -1 is only a placeholder for the local vLLM server, not a real key.

## Install
```
pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
pip install transformer_lens transformers scikit-learn matplotlib numpy google-genai
# phase -1 only:
pip install vllm inspect-ai "inspect-evals[agentic_misalignment]" openai   # openai is just the client library for the local server
```
(The torch pin is taken from your phase 2 notebook; keep it if TransformerLens complains about versions.)

## Hardware
Gemma 3 12B in bf16 needs about 24 GB of GPU memory (phases 0, 1, 3, 4, 5; phase -1 serves it with vLLM). Phase 2 runs on CPU.

## Order and what feeds what
```
phase -1  runs/*.eval  --(hand-check, already done)-->  contrast_transcripts.json
phase 0   tooling check (no output)
phase 1   contrast_transcripts.json  ->  contrast_acts.pt, contrast_acts_commit.pt
phase 2   contrast_acts*.pt          ->  commit_diff_means.pt, probe plots
phase 3   commit_diff_means.pt       ->  results/phase3sweep_*
phase 4   commit_diff_means.pt       ->  results/phase4_*
phase 5   (own direction)            ->  desperation_dir.pt, results/phase5_*
```
You can skip phases -1 to 2 and go straight to 3-5 with the two shipped files.

## One thing to know about grading
The archived numbers in the notebooks came from GPT-4o. The notebooks now grade with Gemini and save to separate `*_graded_gemini.json` files, so a rerun will give slightly different rates. Compare against the quoted numbers with that in mind.
