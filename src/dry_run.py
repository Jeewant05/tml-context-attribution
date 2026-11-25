"""Dry run: plant attacks, generate, check success, run leave-one-out, test recovery.

Usage (on a GPU node, env active):
    python src/dry_run.py --data data/nq/*.jsonl.gz --n 20 --out out/dry_run

Writes out/dry_run/examples.jsonl (one row per attacked example) and prints a summary.
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

# condition name -> (attack, copies)
CONDITIONS = {
    "link": ("link", 1),
    "link_both": ("link_both", 1),
    "link_override": ("link_override", 1),
    "caps": ("caps", 1),
    "two_copy": ("link", 2),
    "two_copy_both": ("link_both", 2),
    "two_copy_override": ("link_override", 2),
}


class Model:
    def __init__(self, path):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, device_map="cuda")
        self.model.eval()
        self.passes = 0  # sequences pushed through the model in scoring

    def _pad_left(self, seqs):
        L = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, L - len(s):] = torch.tensor(s)
            mask[i, L - len(s):] = 1
        return ids.cuda(), mask.cuda()

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=64, bs=8):
        """Greedy replies. Returns list of (text, token_ids without eos/pad)."""
        out = []
        stop = {self.tok.eos_token_id, self.tok.pad_token_id,
                self.tok.convert_tokens_to_ids("<|im_end|>")}
        for b in range(0, len(prompts), bs):
            enc = [self.tok(p, add_special_tokens=False).input_ids for p in prompts[b:b + bs]]
            ids, mask = self._pad_left(enc)
            gen = self.model.generate(input_ids=ids, attention_mask=mask,
                                      max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
            for row in gen[:, ids.shape[1]:].tolist():
                toks = []
                for t in row:
                    if t in stop:
                        break
                    toks.append(t)
                out.append((self.tok.decode(toks, skip_special_tokens=True).strip(), toks))
        return out

    @torch.no_grad()
    def logprob(self, pairs, bs=8):
        """pairs: list of (prompt_text, reply_token_ids). Returns sum log p(reply | prompt)."""
        res = []
        for b in range(0, len(pairs), bs):
            chunk = pairs[b:b + bs]
            p_ids = [self.tok(p, add_special_tokens=False).input_ids for p, _ in chunk]
            seqs = [p + list(r) for p, (_, r) in zip(p_ids, chunk)]
            rl = [len(r) for _, r in chunk]
            ids, mask = self._pad_left(seqs)
            pos = (mask.cumsum(-1) - 1).clamp(min=0)
            K = max(rl) + 1
            logits = self.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                                logits_to_keep=K).logits.float()  # [B, K, V]
            lp = torch.log_softmax(logits[:, :-1], dim=-1)
            L = ids.shape[1]
            tgt = ids[:, L - K + 1:]
            tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [B, K-1]
            for i, r in enumerate(rl):
                res.append(tok_lp[i, K - 1 - r:].sum().item() if r > 0 else 0.0)
            self.passes += len(chunk)
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/common/data/models/qwen--Qwen3-8B")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/dry_run")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--conditions", default="link,caps,two_copy",
                    help="comma list from: " + ",".join(CONDITIONS))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    conds = args.conditions.split(",")
    for c in conds:
        if c not in CONDITIONS:
            raise SystemExit(f"unknown condition {c}")

    t0 = time.time()
    rows = C.load_nq(args.data)
    rng = C.seeded_rng(args.seed)
    sample = rng.sample(rows, args.n)
    print(f"loaded {len(rows)} questions, sampled {len(sample)}", flush=True)

    M = Model(args.model)
    print(f"GPU: {torch.cuda.get_device_name(0)} | model loaded in {time.time() - t0:.0f}s",
          flush=True)
    timings = {}

    # 1) Build clean + attacked prompts.
    jobs = []  # each: dict with prompt and metadata
    for qi, ex in enumerate(sample):
        jobs.append({"qi": qi, "cond": "clean", "planted": [], "passages": ex["passages"]})
        # Same planted positions for every variant of a question, so variants compare fairly.
        pos = {1: C.choose_planted(rng, ex["answer_idx"], 1),
               2: C.choose_planted(rng, ex["answer_idx"], 2)}
        for cname in conds:
            attack, k = CONDITIONS[cname]
            planted = pos[k]
            jobs.append({"qi": qi, "cond": cname, "attack": attack,
                         "planted": planted, "passages": C.plant(ex["passages"], planted, attack)})
    for j in jobs:
        j["prompt"] = C.build_prompt(M.tok, sample[j["qi"]]["question"], j["passages"])

    # 2) Generate.
    t = time.time()
    gens = M.generate([j["prompt"] for j in jobs], bs=args.bs)
    timings["generate_s"] = time.time() - t
    for j, (text, ids) in zip(jobs, gens):
        j["reply"], j["reply_ids"] = text, ids
    clean = {j["qi"]: j for j in jobs if j["cond"] == "clean"}

    # 3) For successful attacks: margin for full context and each leave-one-out subset.
    t = time.time()
    results = []
    for j in jobs:
        if j["cond"] == "clean":
            continue
        ex = sample[j["qi"]]
        c = clean[j["qi"]]
        ok = C.attack_succeeded(j["attack"], j["reply"])
        row = {"qi": j["qi"], "question": ex["question"], "answers": ex["answers"],
               "source": ex["source"], "cond": j["cond"], "answer_idx": ex["answer_idx"],
               "planted": j["planted"], "clean_reply": c["reply"],
               "clean_correct": C.answer_correct(c["reply"], ex["answers"]),
               "attacked_reply": j["reply"], "success": ok}
        # An attack that leaves the reply unchanged has no margin to explain.
        if ok and j["reply_ids"] != c["reply_ids"] and j["reply_ids"] and c["reply_ids"]:
            subsets = [list(range(10))] + [[i for i in range(10) if i != d] for d in range(10)]
            prompts = [C.build_prompt(M.tok, ex["question"], [j["passages"][i] for i in s])
                       for s in subsets]
            pairs = [(p, j["reply_ids"]) for p in prompts] + [(p, c["reply_ids"]) for p in prompts]
            lps = M.logprob(pairs, bs=args.bs)
            att, cln = lps[:11], lps[11:]
            margins = [a - b for a, b in zip(att, cln)]
            full = margins[0]
            loo = [full - margins[d + 1] for d in range(10)]  # drop in margin when d removed
            overlap = C.word_overlap_scores(j["reply"], j["passages"])
            rank_loo = sorted(range(10), key=lambda i: -loo[i])
            rank_ov = sorted(range(10), key=lambda i: -overlap[i])
            k = len(j["planted"])
            row.update({
                "margin_full": full, "loo_drop": loo, "overlap": overlap,
                "loo_hit": set(rank_loo[:k]) == set(j["planted"]),
                "overlap_hit": set(rank_ov[:k]) == set(j["planted"]),
                "loo_top": rank_loo[:3], "overlap_top": rank_ov[:3],
                "planted_drop": [loo[i] for i in j["planted"]],
            })
            # Recovery: remove the planted passage(s) and regenerate.
            keep = [i for i in range(10) if i not in j["planted"]]
            rec_prompt = C.build_prompt(M.tok, ex["question"], [j["passages"][i] for i in keep])
            rec_reply = M.generate([rec_prompt])[0][0]
            row["recovery_reply"] = rec_reply
            row["recovered"] = not C.attack_succeeded(j["attack"], rec_reply)
            if k == 2:  # remove only the top-LOO copy
                one = [i for i in range(10) if i != rank_loo[0]]
                p1 = C.build_prompt(M.tok, ex["question"], [j["passages"][i] for i in one])
                r1 = M.generate([p1])[0][0]
                row["recovered_one_removed"] = not C.attack_succeeded(j["attack"], r1)
        results.append(row)
        print(f"[{j['qi']:02d} {j['cond']:8s}] success={ok} "
              f"loo_hit={row.get('loo_hit')} reply={j['reply'][:70]!r}", flush=True)
    timings["attribute_s"] = time.time() - t

    with open(os.path.join(args.out, "examples.jsonl"), "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    # 4) Summary.
    lines = [f"GPU: {torch.cuda.get_device_name(0)}",
             f"Questions: {len(sample)} | clean accuracy: "
             f"{sum(r['clean_correct'] for r in results if r['cond'] == conds[0])}/{len(sample)}"]
    for cond in conds:
        rs = [r for r in results if r["cond"] == cond]
        sc = [r for r in rs if "loo_hit" in r]
        lines.append(f"{cond:9s} attack success {sum(r['success'] for r in rs)}/{len(rs)} | "
                     f"LOO top-k hit {sum(r['loo_hit'] for r in sc)}/{len(sc)} | "
                     f"overlap hit {sum(r['overlap_hit'] for r in sc)}/{len(sc)} | "
                     f"recovered {sum(r['recovered'] for r in sc)}/{len(sc)}")
        hits = {}
        for r in rs:
            for i in r["planted"]:
                h = hits.setdefault(i, [0, 0]); h[1] += 1; h[0] += r["success"]
        lines.append("          success by planted position (1-10): " + " ".join(
            f"{i + 1}:{h[0]}/{h[1]}" for i, h in sorted(hits.items())))
        if CONDITIONS[cond][1] == 2 and sc:
            drops = [d for r in sc for d in r["planted_drop"]]
            lines.append(f"          two-copy median single-removal margin drop: "
                         f"{statistics.median(drops):.2f} | recovered removing only top-1: "
                         f"{sum(r['recovered_one_removed'] for r in sc)}/{len(sc)}")
        if CONDITIONS[cond][1] == 1 and sc:
            drops = [d for r in sc for d in r["planted_drop"]]
            lines.append(f"          one-copy median margin drop: {statistics.median(drops):.2f}")
    lines.append(f"Scoring sequences: {M.passes} | generate {timings['generate_s']:.0f}s | "
                 f"attribution+recovery {timings['attribute_s']:.0f}s | "
                 f"total {time.time() - t0:.0f}s | peak mem "
                 f"{torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    summary = "\n".join(lines)
    print("\n=== SUMMARY ===\n" + summary)
    with open(os.path.join(args.out, "summary.txt"), "w") as fh:
        fh.write(summary + "\n")


if __name__ == "__main__":
    main()
