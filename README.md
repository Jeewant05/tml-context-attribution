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
data/nq/           the three 10-passage NQ-Open files (source data, see below)
data/              prompts.jsonl (Dominic). Other raw downloads go in data/raw/ (not committed).
src/               scripts: generate.py, attribute.py, evaluate.py, ...
outputs/           generations.jsonl, recovery.jsonl (Jeewant)
outputs/scores/    one {method}.jsonl per attribution method
docs/              project plan
```

## Source data: `data/nq/`

The three 10-passage files from the [Lost in the Middle](https://github.com/nelson-liu/lost-in-the-middle) release (`qa_data/10_total_documents/`). Checked on Nov 24:

- 2,655 rows each, 10 passages per row, same questions in the same order across the three files
- the `isgold` passage sits at index 0, 4 and 9 respectively
- the gold passage always has `hasanswer = true`, and no other passage ever does
- row keys: `question`, `answers`, `ctxs`, `nq_annotated_gold`. Passage keys: `title`, `text`, `isgold`, `hasanswer` (`gold_at_4` and `gold_at_9` also have `id`, `score`, `original_retrieval_index`)

| File | SHA-256 |
| --- | --- |
| `nq-open-10_total_documents_gold_at_0.jsonl.gz` | `192a05b27af2b09eec33ca0c94bb5cf82bcaf70d78b3bdff1258df34bf37aab9` |
| `nq-open-10_total_documents_gold_at_4.jsonl.gz` | `f8425253dafc19cd9de31c06ab6c8b57d4da2bf48b8ea2d0790b805fb3aa1968` |
| `nq-open-10_total_documents_gold_at_9.jsonl.gz` | `fab97badc6d69bae53c805bb198e8742a78ddf3c40a076989ef62891b76c997f` |

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
| `titles` | list[str] | 10 passage titles. The Lost in the Middle prompt formats each passage as `Document [i](Title: {title}) {text}` (1-based `i`), so generation needs them |
| `passages` | list[str] | 10 passage texts (no title), attack text already appended |

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

Build the environment inside an `interact` session on an A100 node, not on the login node:

```bash
module reset; module load Miniforge3
conda create -p ~/.conda/envs/tml_a100 python=3.12
source activate ~/.conda/envs/tml_a100
pip install torch "transformers>=4.51" accelerate sentence-transformers pandas tqdm
```

- Model: `/common/data/models/qwen--Qwen3-8B` (shared copy on ARC, no download needed).
- Slurm account `tml_2026`, partition `a100_normal_q` (see `jobs/*.sbatch`).
- GPU type and hours for every job are in `docs/compute_log.md` (raw sacct lines in `jobs_log.csv`).

## Rules for the repo

- Each shared file has one owner (tables above). Don't change someone else's file format without posting in the group chat first.
- Never commit model weights, the Hugging Face cache, raw downloads or Slurm logs (see `.gitignore`). GitHub rejects files over 100 MB; keep anything big on ARC.

## License

Code and docs in this repo are under the [MIT License](LICENSE).

The files in `data/nq/` are not ours and keep their own terms. They come from the [Lost in the Middle](https://github.com/nelson-liu/lost-in-the-middle) release (MIT), built on [NQ-Open](https://github.com/google-research-datasets/natural-questions/tree/master/nq_open) (CC BY-SA 3.0) and English Wikipedia passages (CC BY-SA). If you reuse them, credit those sources and share changes to the data under CC BY-SA. The same applies to `data/prompts.jsonl`, which is built from them.

## Status

- [x] Plan and shared file formats
- [x] NQ-Open 10-passage files downloaded and checked
- [ ] Dry run on Qwen3-8B, 20 questions (Nov 24)
- [ ] `data/prompts.jsonl` (Dominic)
- [ ] Scoring function tested on 5 hand-made cases (Yash)
- [x] Pilot on 50 questions with Qwen3-8B, go/no-go per attack (Jeewant)
- [x] Full generation: 2,400 rows, link 83%, caps 99%, two_copy 86% (Nov 25)
- [ ] Full generation, attribution, recovery
- [ ] `evaluate.py`, figures, report, slides (Adli)
