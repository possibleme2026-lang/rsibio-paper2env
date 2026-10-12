#!/usr/bin/env python3
"""Force /logs/verifier/reward.json to be numeric-only.

Harbor type-checks the reward file as ``dict[str, float | int]``; a nested
object or string anywhere in it makes pydantic abort the trial *after* the
agent has finished, so the rollout is recorded as an exception rather than a
score. This step runs after the grader and moves anything non-numeric into
/logs/verifier/report.json, leaving a valid reward.json behind.

Also covers the "grader died before writing anything" case: it synthesises
{"score": 0.0, "reward": 0.0} so the trial reports a real zero instead of a
RewardFileNotFoundError exception.
"""
import json
import os

REWARD = "/logs/verifier/reward.json"
REPORT = "/logs/verifier/report.json"


def _numeric(v):
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    return None


def main():
    os.makedirs("/logs/verifier", exist_ok=True)
    payload = {}
    if os.path.exists(REWARD):
        try:
            payload = json.load(open(REWARD))
        except Exception:
            payload = {}
    if not isinstance(payload, dict):
        payload = {"score": payload, "reward": payload}

    score = _numeric(payload.get("score"))
    reward = _numeric(payload.get("reward"))
    if reward is None:
        reward = score
    if score is None:
        score = reward
    if score is None:
        score = reward = 0.0

    # keep every scalar entry (no information loss for numeric metrics) and
    # move only the non-numeric ones -- those are what break pydantic.
    clean = {"score": score, "reward": reward}
    extra = {}
    for k, v in payload.items():
        if k in ("score", "reward"):
            continue
        num = _numeric(v)
        if num is None or isinstance(v, (dict, list)):
            extra[k] = v
        else:
            clean[k] = num
    if extra:
        prev = {}
        if os.path.exists(REPORT):
            try:
                prev = json.load(open(REPORT))
            except Exception:
                prev = {}
        if isinstance(prev, dict):
            prev.update(extra)
        else:
            prev = extra
        with open(REPORT, "w") as fh:
            json.dump(prev, fh, indent=2, default=str)

    with open(REWARD, "w") as fh:
        json.dump(clean, fh)
    print("sanitized reward.json ->", clean)


if __name__ == "__main__":
    main()
