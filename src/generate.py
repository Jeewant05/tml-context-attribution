"""Generate replies for every row of data/prompts.jsonl (team shared format).

Writes outputs/generations.jsonl: one row per (id, condition) with
  id, condition, reply, success (null for clean), plus extras:
  reply_ids (token ids, so scoring uses the exact reply), answer_correct, n_new_tokens.
Resumable: rows already in the output file are skipped.

Usage: python src/generate.py --prompts PATH --out outputs/generations.jsonl [--limit N]
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from dry_run import Model  # noqa: E402  (batched greedy generate, left padding)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/common/data/models/qwen--Qwen3-8B")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default="outputs/generations.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="only the first N question ids")
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.prompts)]
    if args.limit:
        keep_ids = list(dict.fromkeys(r["id"] for r in rows))[:args.limit]
        keep_ids = set(keep_ids)
        rows = [r for r in rows if r["id"] in keep_ids]
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            r = json.loads(l)
            done.add((r["id"], r["condition"]))
    todo = [r for r in rows if (r["id"], r["condition"]) not in done]
    print(f"{len(rows)} rows, {len(done)} already done, {len(todo)} to generate", flush=True)
    if not todo:
        return

    t0 = time.time()
    M = Model(args.model)
    print(f"GPU: {torch.cuda.get_device_name(0)} | load {time.time() - t0:.0f}s", flush=True)
    prompts = [C.build_prompt_from_row(M.tok, r) for r in todo]
    # Sort by length so batches pad less; write in batches so a crash loses little.
    order = sorted(range(len(todo)), key=lambda i: len(prompts[i]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t1 = time.time()
    n = 0
    with open(args.out, "a") as fh:
        for b in range(0, len(order), args.bs):
            idx = order[b:b + args.bs]
            outs = M.generate([prompts[i] for i in idx], bs=args.bs)
            for i, (text, ids) in zip(idx, outs):
                r = todo[i]
                fh.write(json.dumps({
                    "id": r["id"], "condition": r["condition"], "reply": text,
                    "success": C.success_for_condition(r["condition"], text),
                    "answer_correct": C.answer_correct(text, r["answers"]),
                    "n_new_tokens": len(ids), "reply_ids": ids,
                }) + "\n")
            fh.flush()
            n += len(idx)
            if (b // args.bs) % 10 == 0:
                print(f"{n}/{len(todo)} done, {time.time() - t1:.0f}s", flush=True)

    # Summary over the whole output file.
    allr = [json.loads(l) for l in open(args.out)]
    print(f"\n=== GENERATION SUMMARY ({len(allr)} rows) ===")
    for cond in ["clean", "link", "caps", "two_copy"]:
        rs = [r for r in allr if r["condition"] == cond]
        if not rs:
            continue
        acc = sum(r["answer_correct"] for r in rs)
        msg = f"{cond:9s} n={len(rs)} answer_correct={acc}"
        if cond != "clean":
            msg += f" success={sum(r['success'] for r in rs)} ({sum(r['success'] for r in rs) / len(rs):.0%})"
        print(msg)
    print(f"gen time {time.time() - t1:.0f}s | peak mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
