"""Solid RL experiment for the paper: train -> held-out eval -> JSON.

Protocol (everything downstream, including figures, reads from the two JSON
artifacts this script writes):

  Split      wave2 papers are a subset of wave1, so the unique pool is 18
             papers. random.Random(123) holds out 4 from wave1 + 2 from
             wave2's ids (disjoint draw) -> 12 training / 6 held-out papers.
             The split is persisted to the eval JSON, so the exact
             train/eval partition is reproducible.
  Train      48 GRPO steps x 3 prompts x group 4, temperature 0.9,
             max_tokens 768, lr 1e-4, LoRA rank 32, importance_sampling loss.
             Zero-variance groups are skipped (no gradient), as in
             rl_tinker_train.py.
  Eval       base model vs final RL sampler, both via fresh sampling clients,
             temperature 0.3 (near-greedy but stochastic enough to see the
             distribution), max_tokens 900, n=5 samples per paper x 6 papers
             = 30 completions per arm. Reward is the SAME deterministic
             validator stack used in training (schema / emit+gates /
             provenance fidelity) - verifiable, no judge model.
  Stats      per-(arm, paper) means; paired bootstrap over the 6 papers
             (10 000 resamples, seed 42) for the delta of each metric with
             95% CI.

Outputs
  out/rl_runs/experiment_log.json    per-step training telemetry
  out/rl_runs/experiment_eval.json   split, per-sample records, per-paper
                                     means, bootstrap CIs

Usage:
  TINKER_API_KEY=... .rlvenv/Scripts/python.exe tools/rl_experiment.py \
      --steps 48 --skip-train-if-log-exists
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from rl_tinker_train import (  # noqa: E402
    BASE_MODEL, CORPUS, build_datum, build_prompt, reward_fn,
)

import tinker  # noqa: E402
from tinker import types  # noqa: E402

LOG_DIR = REPO / "out" / "rl_runs"
TRAIN_LOG = LOG_DIR / "experiment_log.json"
EVAL_LOG = LOG_DIR / "experiment_eval.json"

HELD_OUT_WAVE1 = 4
HELD_OUT_WAVE2 = 2
SPLIT_SEED = 123
BOOT_SEED = 42
BOOT_N = 10_000

METRICS = ("reward", "schema_ok", "gates_ok", "fidelity")


# --------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------

def make_split():
    wave1 = json.load(open(CORPUS, encoding="utf-8"))["records"]
    wave2 = json.load(open(REPO / "out" / "corpus_wave2.json",
                           encoding="utf-8"))["records"]
    by_id = {}
    for r in wave1 + wave2:
        by_id.setdefault(r["doi"], r)
    rng = random.Random(SPLIT_SEED)
    w1_ids = sorted({r["doi"] for r in wave1})
    w2_ids = sorted({r["doi"] for r in wave2})
    held1 = set(rng.sample(w1_ids, HELD_OUT_WAVE1))
    held2 = set(rng.sample(w2_ids, HELD_OUT_WAVE2))
    held_out = held1 | held2
    train_ids = sorted((set(w1_ids) | set(w2_ids)) - held_out)
    return ([by_id[i] for i in train_ids],
            [by_id[i] for i in sorted(held_out)])


# --------------------------------------------------------------------------
# training (GRPO, same loop as rl_tinker_train but richer telemetry)
# --------------------------------------------------------------------------

def train(args, train_recs, rng) -> dict:
    service = tinker.ServiceClient()
    training_client = service.create_lora_training_client(
        base_model=BASE_MODEL, rank=args.lora_rank)
    tokenizer = training_client.get_tokenizer()
    print(f"[tinker] base={BASE_MODEL} lora_rank={args.lora_rank}", flush=True)

    log = {"base_model": BASE_MODEL, "steps": [], "config": vars(args)}
    t0 = time.time()
    for step in range(1, args.steps + 1):
        sampling_client = training_client.save_weights_and_get_sampling_client()
        batch = rng.sample(train_recs, min(args.prompts_per_step, len(train_recs)))
        datums: list[types.Datum] = []
        rewards: list[float] = []
        n_schema = n_gates = 0
        n_groups_kept = 0

        for rec in batch:
            prompt = build_prompt(rec)
            prompt_tokens = tokenizer.encode(prompt)
            params = types.SamplingParams(
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=0.95,
                seed=rng.randint(0, 2**31 - 1),
            )
            resp = sampling_client.sample(
                prompt=types.ModelInput.from_ints(prompt_tokens),
                num_samples=args.group_size,
                sampling_params=params,
            ).result()
            group = []
            for seq in resp.sequences:
                text = tokenizer.decode(seq.tokens)
                r, detail = reward_fn(text, rec)
                group.append((seq, r))
                rewards.append(r)
                if detail.get("schema"):
                    n_schema += 1
                if detail.get("gates") in ("ok", "warning"):
                    n_gates += 1
            vals = [r for _, r in group]
            if max(vals) - min(vals) < 1e-9:
                continue
            mean_r = sum(vals) / len(vals)
            n_groups_kept += 1
            for seq, r in group:
                datums.append(build_datum(
                    tokenizer, prompt_tokens, list(seq.tokens),
                    list(seq.logprobs), r - mean_r))

        if datums:
            fb = training_client.forward_backward(
                data=datums, loss_fn="importance_sampling")
            opt = training_client.optim_step(
                types.AdamParams(learning_rate=args.lr))
            fb.result()
            opt.result()

        n = len(rewards)
        rec_step = {
            "step": step,
            "mean_reward": round(sum(rewards) / n, 4) if n else 0.0,
            "schema_rate": round(n_schema / n, 4) if n else 0.0,
            "gates_rate": round(n_gates / n, 4) if n else 0.0,
            "groups_kept": n_groups_kept,
            "n_datums": len(datums),
            "max_reward": max(rewards) if rewards else 0.0,
            "elapsed_sec": round(time.time() - t0, 1),
        }
        log["steps"].append(rec_step)
        print(json.dumps(rec_step), flush=True)
        # flush telemetry every 4 steps so a crash still leaves a curve
        if step % 4 == 0 or step == args.steps:
            TRAIN_LOG.write_text(
                json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    save = training_client.save_state(name="rsibio-rl-exp-final")
    sampler = training_client.save_weights_for_sampler(name="rsibio-rl-exp-sampler")
    log["final_training_checkpoint"] = str(save.result().path)
    log["final_sampler_checkpoint"] = str(sampler.result().path)
    TRAIN_LOG.write_text(json.dumps(log, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"[train done] sampler ckpt: {log['final_sampler_checkpoint']}",
          flush=True)
    return log


# --------------------------------------------------------------------------
# evaluation: base vs RL sampler on held-out papers
# --------------------------------------------------------------------------

def eval_arm(service, sampling_client, tokenizer, held_recs, n_samples,
             temperature, max_tokens, rng, arm_name):
    per_sample = []
    for rec in held_recs:
        prompt = build_prompt(rec)
        prompt_tokens = tokenizer.encode(prompt)
        params = types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            seed=rng.randint(0, 2**31 - 1),
        )
        resp = sampling_client.sample(
            prompt=types.ModelInput.from_ints(prompt_tokens),
            num_samples=n_samples,
            sampling_params=params,
        ).result()
        for seq in resp.sequences:
            text = tokenizer.decode(seq.tokens)
            r, detail = reward_fn(text, rec)
            per_sample.append({
                "paper": rec["doi"],
                "reward": round(r, 4),
                "schema_ok": bool(detail.get("schema")),
                "gates_ok": detail.get("gates") in ("ok", "warning"),
                "fidelity": float(detail.get("fidelity", 0.0)),
                "error": detail.get("error", ""),
            })
        print(f"[eval:{arm_name}] {rec['doi']} done "
              f"({sum(1 for _ in per_sample)} total)", flush=True)
    return per_sample


def per_paper_means(per_sample):
    papers = sorted({s["paper"] for s in per_sample})
    out = {}
    for p in papers:
        rows = [s for s in per_sample if s["paper"] == p]
        out[p] = {m: sum(s[m] for s in rows) / len(rows)
                  for m in METRICS}
    return out


def paired_bootstrap(pp_a, pp_b, seed=BOOT_SEED, n_boot=BOOT_N):
    """Bootstrap over papers: mean metric delta (rl - base) with 95% CI."""
    papers = sorted(pp_a)
    rng = random.Random(seed)
    res = {}
    for m in METRICS:
        deltas = [pp_b[p][m] - pp_a[p][m] for p in papers]
        boots = []
        for _ in range(n_boot):
            idx = [rng.randrange(len(papers)) for _ in papers]
            boots.append(sum(deltas[i] for i in idx) / len(idx))
        boots.sort()
        lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]
        res[m] = {
            "mean_delta": round(sum(deltas) / len(deltas), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "significant": not (lo <= 0.0 <= hi),
            "per_paper_delta": {p: round(d, 4)
                                for p, d in zip(papers, deltas)},
        }
    return res


def evaluate(log, held_recs, args) -> dict:
    service = tinker.ServiceClient()
    ckpt = log["final_sampler_checkpoint"]

    base_client = service.create_sampling_client(base_model=BASE_MODEL)
    rl_client = service.create_sampling_client(model_path=ckpt)
    base_tok = base_client.get_tokenizer()
    rl_tok = rl_client.get_tokenizer()

    rng = random.Random(args.seed + 1)
    print("[eval] arm=base", flush=True)
    base_samples = eval_arm(service, base_client, base_tok, held_recs,
                            args.eval_samples, args.eval_temperature,
                            args.eval_max_tokens, rng, "base")
    print("[eval] arm=rl", flush=True)
    rl_samples = eval_arm(service, rl_client, rl_tok, held_recs,
                          args.eval_samples, args.eval_temperature,
                          args.eval_max_tokens, rng, "rl")

    pp_base, pp_rl = per_paper_means(base_samples), per_paper_means(rl_samples)
    out = {
        "base_model": BASE_MODEL,
        "rl_checkpoint": ckpt,
        "config": {
            "eval_temperature": args.eval_temperature,
            "eval_max_tokens": args.eval_max_tokens,
            "eval_samples_per_paper": args.eval_samples,
            "n_held_out_papers": len(held_recs),
        },
        "split": {
            "seed": SPLIT_SEED,
            "held_out": [r["doi"] for r in held_recs],
            "held_out_titles": {r["doi"]: r.get("title", "")
                                for r in held_recs},
        },
        "per_sample": {"base": base_samples, "rl": rl_samples},
        "per_paper_means": {"base": pp_base, "rl": pp_rl},
        "summary": {},
        "paired_bootstrap_rl_minus_base": {},
    }
    for m in METRICS:
        out["summary"][m] = {
            "base": round(sum(v[m] for v in pp_base.values()) / len(pp_base), 4),
            "rl": round(sum(v[m] for v in pp_rl.values()) / len(pp_rl), 4),
        }
    out["paired_bootstrap_rl_minus_base"] = paired_bootstrap(pp_base, pp_rl)
    return out


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--prompts-per-step", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--eval-samples", type=int, default=5)
    ap.add_argument("--eval-temperature", type=float, default=0.3)
    ap.add_argument("--eval-max-tokens", type=int, default=900)
    ap.add_argument("--skip-train", action="store_true",
                    help="reuse existing checkpoint from experiment_log.json")
    args = ap.parse_args(argv)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    train_recs, held_recs = make_split()
    print(f"[split] train={len(train_recs)} held_out={len(held_recs)}",
          flush=True)

    if args.skip_train and TRAIN_LOG.exists():
        log = json.loads(TRAIN_LOG.read_text(encoding="utf-8"))
        print(f"[skip-train] reusing ckpt {log['final_sampler_checkpoint']}",
              flush=True)
    else:
        rng = random.Random(args.seed)
        log = train(args, train_recs, rng)

    ev = evaluate(log, held_recs, args)
    EVAL_LOG.write_text(json.dumps(ev, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[done] eval -> {EVAL_LOG}", flush=True)
    print(json.dumps(ev["summary"], indent=2), flush=True)
    print(json.dumps(ev["paired_bootstrap_rl_minus_base"], indent=2),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
