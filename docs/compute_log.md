# Compute log (ARC TinkerCliffs)

All jobs ran on account `tml_2026`, partition `a100_normal_q`, 1 GPU each (NVIDIA A100-SXM4-80GB).
Wall time is sacct `Elapsed`, so GPU-hours = wall time × 1 GPU. Raw sacct lines are in `jobs_log.csv`.

| Job ID | Purpose | GPU | Wall time | GPU-hours | Peak GPU memory |
| --- | --- | --- | --- | --- | --- |
| 8978364 | Environment build + smoke test (interactive) | A100-SXM4-80GB | 00:21:21 | 0.356 | 17.1 GB |
| 8979203 | Dry run: 20 questions, link / caps / two_copy | A100-SXM4-80GB | 00:04:14 | 0.071 | 20.7 GB |
| 8995768 | Pilot: 50 questions (seed 1), 7 attack conditions | A100-SXM4-80GB | 00:21:11 | 0.353 | 21.2 GB |
| 8996299 | Full generation: 2,400 rows of `data/prompts.jsonl` | A100-SXM4-80GB | 00:14:44 | 0.246 | 24.8 GB |
| **Total** | 4 jobs | | **01:01:30** | **1.025** | max 24.8 GB |

Peak memory is PyTorch's `max_memory_allocated`, as printed by each script. The smoke-test figure
is one forward pass at about 1,500 tokens (measured Nov 24).
