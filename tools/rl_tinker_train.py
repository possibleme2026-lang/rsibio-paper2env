"""RL post-training on tinker: GRPO-style loop, NO SFT stage.

Trains the cheapest capable tinker model (openai/gpt-oss-20b, $0.396/M train
tokens) on the rsibio-paper2env synthesized environment set.

Task: given a paper title + abstract (from the corpus), the model must output
a TaskBlueprint-shaped JSON. Reward is the SAME four-gate discipline the
pipeline enforces offline, compressed into three graded components:

  r_schema  (0.30)  strict blueprint_1 JSON schema validation
  r_gates   (0.45)  emit + four-gate verify_task_dir pass
  r_quality (0.25)  provenance fidelity: DOI/paper_id copied from the input
                    block, metric named, tests/test.sh present

This is a verifiable-reward environment (the paper's Stage-1 story in
miniature): reward comes from deterministic validators, not a judge model.

Loop (importance_sampling loss, GRPO-style group centering):
  for each step:
    sample G completions per prompt from the current policy
    reward each completion with the offline validators
    skip groups with zero variance (no signal)
    build Datums: shift-by-one, advantages on completion tokens only
    forward_backward(loss_fn="importance_sampling") + optim_step(Adam)

Usage (this venv has tinker 0.33.1):
  TINKER_API_KEY=... .rlvenv/Scripts/python.exe tools/rl_tinker_train.py \
      --steps 6 --group-size 4 --prompts-per-step 2
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import tinker  # noqa: E402
from tinker import types  # noqa: E402

BASE_MODEL = "openai/gpt-oss-20b"          # cheapest trainable ($0.396/M)
CORPUS = REPO / "out" / "corpus_biorxiv_2026-09.json"
LOG_PATH = REPO / "out" / "rl_runs"

# --------------------------------------------------------------------------
# prompts / reward
# --------------------------------------------------------------------------

SYSTEM_tmpl = """You are the Construction agent of rsibio-paper2env. Given a
life-science paper block, output ONE JSON object (no prose, no markdown
fences) that is a TaskBlueprint. Keys: task_name ("bioforge/<kebab-slug>"),
version "0.1.0", instruction_md, dockerfile (starts FROM python:3.11-slim),
test_files with "tests/test.sh", solution_files, env_files, task_toml with
schema_version "1.4" and task.name == task_name, provenance, category
"computational-biology", difficulty "medium", keywords, metric, notes.
provenance MUST copy paper_id, title, doi VERBATIM from the input block.
test_files, solution_files, env_files are OBJECTS mapping relative path to
file content (strings), e.g. {"environment/data/x.csv": "a,b\\n1,2\\n"} —
use {} when empty, never [].
OUTPUT BUDGET: think for at most ~3 sentences, then emit the JSON. Keep every
string value under 200 characters; test.sh under 10 lines. keywords, notes and
task_toml values keep their types: keywords and notes are ARRAYS of strings.
Finish the JSON completely — an unfinished object scores zero."""

PAPER_BLOCK = """paper_id: {paper_id}
Title: {title}
DOI: {doi}
PMCID: {pmcid}
Year: {year}
Abstract: {abstract}
"""


def build_prompt(rec: dict) -> str:
    return SYSTEM_tmpl + "\n\n" + PAPER_BLOCK.format(
        paper_id=rec.get("doi", ""),
        title=rec.get("title", ""), doi=rec.get("doi", ""),
        pmcid=rec.get("pmcid", "") or "", year=rec.get("pub_year", ""),
        abstract=(rec.get("abstract", "") or "")[:1500],
    )


def _extract_json(text: str):
    """Extract the LAST balanced JSON object from a completion.

    Reasoning models narrate JSON fragments inside their CoT (e.g.
    ``keywords: [...]``), and the shared first-match scanner in
    backends/jsonx.py grabs whichever candidate opens first — for two of the
    first three smoke samples that was a bare keyword array, not the
    blueprint. The blueprint is the final, complete object, so we scan
    backwards from the last '}'. Falls back to the shared scanner.
    """
    from bioforge_harbor.backends.jsonx import extract_json
    BS = "\\"
    end = text.rfind("}")
    if end >= 0:
        depth = 0
        in_str = False
        esc = False
        for i in range(end, -1, -1):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == BS:
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "}":
                depth += 1
            elif ch == "{":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[i:end + 1])
                        if isinstance(obj, dict):
                            return obj
                    except Exception:  # noqa: BLE001
                        pass
                    break
    try:
        return extract_json(text)
    except Exception:  # noqa: BLE001
        return None


def reward_fn(completion_text: str, rec: dict) -> tuple[float, dict]:
    """Deterministic verifiable reward: 3 components, total in [0,1].

    Partial credit on r_schema is deliberate: with an all-or-nothing schema
    gate, a policy that is one field away from valid produces zero-variance
    groups (every sample 0.0), GRPO gets no gradient, and training is a
    no-op. Distinguishing "JSON parses + key set correct" (0.10) from
    "fully schema-valid" (0.30) creates within-group variance near the
    decision boundary.
    """
    detail = {}
    obj = _extract_json(completion_text)
    if obj is None or not isinstance(obj, dict):
        detail["error"] = "no JSON object"
        return 0.0, detail

    from bioforge_harbor.schemas import default_validator
    v = default_validator().validate("blueprint_1", obj)
    if v.ok:
        r_schema = 0.30
    else:
        required = {"task_name", "instruction_md", "dockerfile",
                    "test_files", "provenance"}
        r_schema = 0.10 if required.issubset(obj.keys()) else 0.0
    detail["schema"] = v.ok
    if not v.ok:
        return r_schema, detail

    from bioforge_harbor.models.blueprint import TaskBlueprint
    from bioforge_harbor.harbor_emit import emit_task, verify_task_dir
    try:
        bp = TaskBlueprint.from_dict(obj)
        errs = bp.validate()
    except Exception as e:  # noqa: BLE001
        detail["blueprint"] = str(e)[:120]
        return r_schema, detail
    if errs:
        detail["validate"] = "; ".join(errs)[:200]
        return r_schema, detail

    tmp = Path(tempfile.mkdtemp(prefix="rltask_"))
    try:
        rep = emit_task(bp, tmp / "task")
        ver = verify_task_dir(tmp / "task", run_harbor_gate=False)
        gates_ok = ver.status in ("ok", "warning")
        r_gates = 0.45 if (rep.status != "failed" and gates_ok) else 0.0
        detail["emit"] = rep.status
        detail["gates"] = ver.status
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # provenance fidelity + test entrypoint + named metric
    prov = obj.get("provenance") or {}
    fidelity = 0.0
    if prov.get("paper_id") == rec.get("doi"):
        fidelity += 0.5
    if prov.get("doi") == rec.get("doi"):
        fidelity += 0.25
    if (obj.get("metric") or "").strip():
        fidelity += 0.15
    if "tests/test.sh" in (obj.get("test_files") or {}):
        fidelity += 0.10
    r_quality = 0.25 * min(fidelity / 1.0, 1.0)
    detail["fidelity"] = round(fidelity, 2)

    total = r_schema + r_gates + r_quality
    return total, detail


# --------------------------------------------------------------------------
# data plumbing
# --------------------------------------------------------------------------

def build_datum(tokenizer, prompt_tokens: list[int], completion_tokens: list[int],
                logprobs: list[float], advantage: float) -> types.Datum:
    n_prefix = len(prompt_tokens) - 1
    full = prompt_tokens + completion_tokens
    return types.Datum(
        model_input=types.ModelInput.from_ints(full[:-1]),
        loss_fn_inputs={
            "target_tokens": full[1:],
            "logprobs": [0.0] * n_prefix + list(logprobs),
            "advantages": [0.0] * n_prefix + [advantage] * len(completion_tokens),
        },
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--prompts-per-step", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--dump-dir", default=None,
                    help="write every completion + reward detail here (debug)")
    args = ap.parse_args(argv)

    random.seed(args.seed)
    LOG_PATH.mkdir(parents=True, exist_ok=True)
    records = json.load(open(CORPUS, encoding="utf-8"))["records"]
    rng = random.Random(args.seed)

    service = tinker.ServiceClient()
    training_client = service.create_lora_training_client(
        base_model=BASE_MODEL, rank=args.lora_rank)
    tokenizer = training_client.get_tokenizer()
    print(f"[tinker] base={BASE_MODEL} lora_rank={args.lora_rank}", flush=True)

    log = {"base_model": BASE_MODEL, "steps": [], "config": vars(args)}
    t0 = time.time()
    for step in range(1, args.steps + 1):
        # -- sampling client refreshes each step (on-policy) ----------------
        sampling_client = training_client.save_weights_and_get_sampling_client()
        batch_prompts = rng.sample(records, min(args.prompts_per_step, len(records)))
        datums: list[types.Datum] = []
        step_rewards: list[float] = []
        n_groups_kept = 0

        for rec in batch_prompts:
            prompt = build_prompt(rec)
            prompt_tokens = tokenizer.encode(prompt)
            model_input = types.ModelInput.from_ints(prompt_tokens)
            params = types.SamplingParams(
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=0.95,
                seed=rng.randint(0, 2**31 - 1),
            )
            resp = sampling_client.sample(
                prompt=model_input,
                num_samples=args.group_size,
                sampling_params=params,
            ).result()  # sync API returns a Future
            seqs = resp.sequences
            group = []
            for seq in seqs:
                text = tokenizer.decode(seq.tokens)
                r, detail = reward_fn(text, rec)
                group.append((seq, r, detail))
                step_rewards.append(r)
                if args.dump_dir and not Path(args.dump_dir).joinpath(
                        f"s{step:03d}_{len(group):02d}.txt").exists():
                    Path(args.dump_dir).mkdir(parents=True, exist_ok=True)
                    Path(args.dump_dir, f"s{step:03d}_{len(group):02d}.txt").write_text(
                        f"REWARD={r} DETAIL={json.dumps(detail)}\n---\n{text}",
                        encoding="utf-8")

            rewards = [r for _, r, _ in group]
            if max(rewards) - min(rewards) < 1e-9:
                continue  # zero-variance group: no learning signal
            mean_r = sum(rewards) / len(rewards)
            n_groups_kept += 1
            for seq, r, _det in group:
                adv = r - mean_r
                datums.append(build_datum(
                    tokenizer, prompt_tokens, list(seq.tokens),
                    list(seq.logprobs), adv))

        if datums:
            fb = training_client.forward_backward(
                data=datums, loss_fn="importance_sampling")
            opt = training_client.optim_step(
                types.AdamParams(learning_rate=args.lr))
            fb.result()
            opt.result()
        mean_r = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
        rec_step = {
            "step": step,
            "n_prompts": len(batch_prompts),
            "n_samples": len(step_rewards),
            "groups_kept": n_groups_kept,
            "mean_reward": round(mean_r, 4),
            "max_reward": max(step_rewards) if step_rewards else 0.0,
            "n_datums": len(datums),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        log["steps"].append(rec_step)
        print(json.dumps(rec_step), flush=True)

    # -- persist ----------------------------------------------------------
    save = training_client.save_state(name="rsibio-rl-final")
    save.result()
    sampler = training_client.save_weights_for_sampler(name="rsibio-rl-sampler")
    sampler.result()
    log["final_training_checkpoint"] = str(save.result().path)
    log["final_sampler_checkpoint"] = str(sampler.result().path)
    out = LOG_PATH / "rl_log.json"
    out.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] log -> {out}", flush=True)
    print(f"[done] sampler ckpt: {log['final_sampler_checkpoint']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
