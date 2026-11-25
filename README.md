# Finding the Poisoned Passage with Context Attribution

TML Fall 2026 final project. We plant attacker text in 1 of the 10 retrieved passages a model reads, check whether it hijacks Qwen3-8B's answer, and test which attribution method ranks the planted passage first: similarity, attention, leave-one-out (LOO), Monte Carlo Shapley, and an attention-shortlist-then-LOO combination.

- Full plan: [`docs/project_plan.md`](docs/project_plan.md) (snapshot of the team doc, Nov 24, 2026)
- Report: 8 pages, NeurIPS format, due **Dec 2, 2026**. Presentation **Dec 3, 2026**.

## Team

| Who | Step | Due |
| --- | --- | --- |
| Dominic | 1. Data and attacks: `data/prompts.jsonl` | Nov 25 |
| Yash | 2. Attribution methods (scoring code) | Nov 25 |
| Jeewant | 3. ARC runs: generation, attribution, recovery | Nov 28 |
| Adli | 4. Evaluation, figures, report, slides | Dec 2 |

## Layout

```
data/              prompts.jsonl (Dominic). Raw NQ downloads go in data/raw/ (not committed).
src/               scripts: generate.py, attribute.py, evaluate.py, ...
outputs/           generations.jsonl, recovery.jsonl (Jeewant)
outputs/scores/    one {method}.jsonl per attribution method
docs/              project plan
```

## Shared file formats

Each step reads and writes these files, so it can be built and tested on its own.

**Conventions**

- JSON Lines (`.jsonl`), UTF-8, one JSON object per line.
- Passage indices are **0-based in files** and 1-based in the report.
- `id` is a string, the same for a question across all its conditions. `(id, condition)` is the key that joins every file.
- `condition` is one of `clean`, `link`, `caps`, `two_copy`.

### `data/prompts.jsonl` (Dominic): one row per question × condition

| Field | Type | Notes |
| --- | --- | --- |
| `id` | str | question id |
| `question` | str | |
| `answers` | list[str] | gold answer strings (any match counts as correct) |
| `gold_idx` | int | index of the `isgold` passage |
| `planted_idx` | list[int] | `[]` for clean, one index for link/caps, two for two_copy; never `gold_idx` |
| `condition` | str | |
| `passages` | list[str] | 10 passages, attack text already appended |

### `outputs/generations.jsonl` (Jeewant): one row per question × condition

| Field | Type | Notes |
| --- | --- | --- |
| `id`, `condition` | str | |
| `reply` | str | greedy, thinking off, ≤64 new tokens |
| `success` | bool or null | attack success check; `null` for clean rows |

### `outputs/scores/{method}.jsonl` (Yash's code, run by Jeewant): one row per successful attack × method

`{method}` is one of `random`, `first`, `similarity`, `attention`, `loo`, `shapley`, `shortlist`.

| Field | Type | Notes |
| --- | --- | --- |
| `id`, `condition` | str | |
| `scores` | list[float] | 10 values; higher = more responsible for the attacked reply |
| `scores_oracle_free` | list[float] or null | 10 values from the attacked reply's log-prob alone; margin methods only (`loo`, `shapley`, `shortlist`), else `null` |
| `passes` | int | forward passes used |
| `gpu_seconds` | float | |

### `outputs/recovery.jsonl` (Jeewant): one row per example × method

| Field | Type | Notes |
| --- | --- | --- |
| `id`, `condition`, `method` | str | |
| `removed_idx` | list[int] | top-1 passage removed (top-2 for two_copy) |
| `reply` | str | reply with those passages removed |
| `attack_gone` | bool | success check now fails |
| `answer_correct` | bool | reply contains any gold answer string |

## Setup (ARC)

```bash
conda create -n tml python=3.11 -y
conda activate tml
pip install -r requirements.txt
```

- Check the shared `/common/data/models/` for Qwen3-8B before downloading it.
- Point `HF_HOME` at a project or scratch directory, not your home directory.
- Log the GPU type and hours for every job (needed for the compute section).

## Rules for the repo

- Each shared file has one owner (tables above). Don't change someone else's file format without posting in the group chat first.
- Never commit model weights, the Hugging Face cache, raw downloads or Slurm logs (see `.gitignore`). GitHub rejects files over 100 MB; keep anything big on ARC.

## Status

- [x] Plan and shared file formats
- [ ] Dry run on Qwen3-0.6B, ~20 questions (Jeewant)
- [ ] `data/prompts.jsonl` (Dominic)
- [ ] Scoring function tested on 5 hand-made cases (Yash)
- [ ] Pilot on 50 questions with Qwen3-8B, go/no-go per attack (Jeewant)
- [ ] Full generation, attribution, recovery
- [ ] `evaluate.py`, figures, report, slides (Adli)
