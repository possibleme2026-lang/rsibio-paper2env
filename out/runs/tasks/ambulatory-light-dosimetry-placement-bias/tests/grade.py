import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import gold_gen

OUTPUT = "/app/output/placement_analysis.json"
REWARD = "/logs/verifier/reward.json"
TOL = 0.01


def load_output():
    if not os.path.exists(OUTPUT):
        return None
    try:
        with open(OUTPUT) as f:
            return json.load(f)
    except Exception:
        return None


def close(a, b, tol=TOL):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def check_schema(out):
    if not isinstance(out, dict):
        return False
    required = [
        "excluded_participant_days",
        "valid_sample_count",
        "context_errors",
        "daily_metric_bias",
        "metrics_exceeding_threshold",
    ]
    if set(out.keys()) != set(required):
        return False
    if not isinstance(out["excluded_participant_days"], list):
        return False
    if not isinstance(out["valid_sample_count"], int):
        return False
    for p in ["chest", "wrist"]:
        if p not in out["context_errors"]:
            return False
        if p not in out["daily_metric_bias"]:
            return False
        if p not in out["metrics_exceeding_threshold"]:
            return False
    return True


def check_excluded(out, gold):
    return sorted(out["excluded_participant_days"]) == sorted(gold["excluded_participant_days"])


def check_valid_count(out, gold):
    return out["valid_sample_count"] == gold["valid_sample_count"]


def check_context(out, gold, placement):
    try:
        for ctx in gold_gen.CONTEXTS:
            o = out["context_errors"][placement][ctx]
            g = gold["context_errors"][placement][ctx]
            if not close(o["median_pct_error"], g["median_pct_error"]):
                return False
            if not close(o["mean_pct_error"], g["mean_pct_error"]):
                return False
        return True
    except Exception:
        return False


def check_daily(out, gold, placement):
    try:
        for metric in gold_gen.METRICS:
            o = out["daily_metric_bias"][placement][metric]
            g = gold["daily_metric_bias"][placement][metric]
            if int(o["n_days"]) != int(g["n_days"]):
                return False
            if not close(o["median_abs_bias"], g["median_abs_bias"]):
                return False
            if not close(o["max_abs_bias"], g["max_abs_bias"]):
                return False
        return True
    except Exception:
        return False


def check_threshold(out, gold, placement):
    try:
        return sorted(out["metrics_exceeding_threshold"][placement]) == sorted(
            gold["metrics_exceeding_threshold"][placement]
        )
    except Exception:
        return False


def main():
    out = load_output()
    if out is None:
        score = 0.0
    elif not check_schema(out):
        score = 0.0
    else:
        gold = gold_gen.compute_gold()
        checks = [
            check_excluded(out, gold),
            check_valid_count(out, gold),
            check_context(out, gold, "chest"),
            check_context(out, gold, "wrist"),
            check_daily(out, gold, "chest"),
            check_daily(out, gold, "wrist"),
            check_threshold(out, gold, "chest"),
            check_threshold(out, gold, "wrist"),
        ]
        score = sum(1 for c in checks if c) / len(checks)
    os.makedirs(os.path.dirname(REWARD), exist_ok=True)
    with open(REWARD, "w") as f:
        json.dump({"score": score, "reward": score}, f)
    print(json.dumps({"score": score, "reward": score}))


if __name__ == "__main__":
    main()
