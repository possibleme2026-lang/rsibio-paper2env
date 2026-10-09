"""Stage 4 tooling: held-out splits + contamination audit (paper 4.4).

Two commands:

  split-holdout      time-based corpus split with a *lock file*: the held-out
                     set is frozen (paper ids + corpus digest bound by SHA256)
                     BEFORE training starts, so it cannot drift to match
                     whatever the training set ended up containing.

  check-contamination  n-gram overlap between held-out texts and the training
                     corpus/environment texts. High overlap => the 'held-out'
                     benchmark leaked => Stage-4 gains are uninterpretable.

The training itself (RLVR / RaR / SFT on Harbor environments) is out of scope
for this framework and is delegated to an external trainer (e.g. slime/Miles/
TRL); what this module guarantees is that the *evaluation contract* is fixed
before any weights move.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .metrics import time_split, ngram_contamination
from ..util.io import sha256_text, write_json, read_json


# --------------------------------------------------------------------------
# split + lock
# --------------------------------------------------------------------------
def split_and_lock(corpus_path: Path | str, *, lock_path: Path | str,
                   train_cutoff: int = 2023, holdout_start: int = 2025,
                   holdout_end: int = 2026) -> dict[str, Any]:
    records = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = records.get("records", [])
    split = time_split(records, train_cutoff=train_cutoff,
                       holdout_start=holdout_start, holdout_end=holdout_end)
    corpus_digest = sha256_text(json.dumps(records, sort_keys=True, ensure_ascii=False))
    lock = {
        "corpus_path": str(corpus_path),
        "corpus_digest": corpus_digest,
        "params": {"train_cutoff": train_cutoff, "holdout_start": holdout_start,
                   "holdout_end": holdout_end},
        "train": sorted(split["train"]),
        "holdout": sorted(split["holdout"]),
        "excluded": sorted(split["excluded"]),
        "locked": True,
    }
    write_json(lock_path, lock)
    return lock


def verify_lock(corpus_path: Path | str, lock_path: Path | str) -> tuple[bool, str]:
    """A lock is only meaningful if it still binds the corpus it locked."""
    lock = read_json(lock_path)
    records = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = records.get("records", [])
    now = sha256_text(json.dumps(records, sort_keys=True, ensure_ascii=False))
    if now != lock.get("corpus_digest"):
        return False, "corpus changed after lock (digest mismatch) — split is stale"
    split = time_split(records, **lock["params"])
    if sorted(split["holdout"]) != sorted(lock["holdout"]):
        return False, "holdout membership drifted from lock"
    return True, "lock valid"


# --------------------------------------------------------------------------
# contamination
# --------------------------------------------------------------------------
def check_contamination(lock_path: Path | str, texts_dir: Path | str, *,
                        ngram: int = 8, threshold: float = 0.05) -> dict[str, Any]:
    """Compare each held-out paper text vs all training texts in *texts_dir*.

    texts_dir layout: one .txt per paper id (e.g. epmc-PMC100.txt). Files whose
    id is in the holdout list are the probes; the rest are training text.
    """
    lock = read_json(lock_path)
    holdout = set(lock["holdout"])
    texts_dir = Path(texts_dir)
    files = {p.stem: p.read_text(encoding="utf-8") for p in texts_dir.glob("*.txt")}
    train_text = "\n".join(t for pid, t in files.items() if pid not in holdout)

    results = []
    leaked = 0
    for pid in sorted(holdout):
        text = files.get(pid)
        if text is None:
            results.append({"paper_id": pid, "overlap": None, "note": "text missing"})
            continue
        ov = ngram_contamination(text, train_text, n=ngram)
        flag = ov > threshold
        leaked += int(flag)
        results.append({"paper_id": pid, "overlap": round(ov, 6), "flagged": flag})
    return {
        "stage": 4,
        "ngram": ngram,
        "threshold": threshold,
        "n_holdout": len(holdout),
        "n_with_text": sum(1 for r in results if r["overlap"] is not None),
        "n_flagged": leaked,
        "max_overlap": max((r["overlap"] for r in results if r["overlap"] is not None),
                           default=None),
        "per_paper": results,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="bioforge stage4",
                                 description="Stage 4: held-out split lock + contamination audit (paper 4.4)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("split-holdout")
    s1.add_argument("--corpus", required=True)
    s1.add_argument("--lock", required=True)
    s1.add_argument("--train-cutoff", type=int, default=2023)
    s1.add_argument("--holdout-window", default="2025..2026")

    s2 = sub.add_parser("check-contamination")
    s2.add_argument("--lock", required=True)
    s2.add_argument("--texts-dir", required=True)
    s2.add_argument("--ngram", type=int, default=8)
    s2.add_argument("--threshold", type=float, default=0.05)
    s2.add_argument("--out", default=None)

    s3 = sub.add_parser("verify-lock")
    s3.add_argument("--corpus", required=True)
    s3.add_argument("--lock", required=True)

    a = ap.parse_args(argv)
    if a.cmd == "split-holdout":
        hs, he = [int(x) for x in a.holdout_window.split("..")]
        lock = split_and_lock(a.corpus, lock_path=a.lock,
                              train_cutoff=a.train_cutoff, holdout_start=hs, holdout_end=he)
        print(json.dumps({k: (len(v) if isinstance(v, list) else v)
                          for k, v in lock.items()}, indent=2))
        return 0
    if a.cmd == "check-contamination":
        rep = check_contamination(a.lock, a.texts_dir, ngram=a.ngram, threshold=a.threshold)
        text = json.dumps(rep, indent=2, ensure_ascii=False) + "\n"
        if a.out:
            Path(a.out).write_text(text, encoding="utf-8")
        print(text)
        return 0 if rep["n_flagged"] == 0 else 1
    if a.cmd == "verify-lock":
        ok, msg = verify_lock(a.corpus, a.lock)
        print(msg)
        return 0 if ok else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
