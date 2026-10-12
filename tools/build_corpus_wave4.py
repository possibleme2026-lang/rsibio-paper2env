"""Fetch a fresh, deduplicated life-science corpus from the bioRxiv/medRxiv API.

Builds ``out/corpus_wave3.json`` with >= 100 computational-biology papers.
Uses the bioRxiv/medRxiv *details* API rather than Europe PMC search because it
returns `license` natively — so every record's open-access claim is
machine-verifiable after the fact, and we never have to assume a licence.

Strategy:
  1. Pull monthly windows across 2025-10 .. 2026-09 from both servers
     (100 records per cursor page, paged until exhausted or a cap).
  2. Keep records whose abstract passes `looks_computational` (method + metric
     vocabulary) and that carry a permissive licence.
  3. Deduplicate by DOI and against wave1/wave2.

Usage:
  .rlvenv/Scripts/python.exe tools/build_corpus_wave3.py --target 110
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "out" / "corpus_wave4.json"
API = "https://api.biorxiv.org/details"

LICENSE_MAP = {
    "cc_by": "CC BY",
    "cc_by_nc_nd": "CC BY-NC-ND",
    "cc_by_nc_sa": "CC BY-NC-SA",
    "cc_by_sa": "CC BY-SA",
    "cc0": "CC0",
    "public_domain": "PUBLIC DOMAIN",
}
OPEN = set(LICENSE_MAP.values())

# Monthly windows. wave1/wave2 already consumed 2026-09, so start at 2026-08.
WINDOWS = [
    ("2025-09-01", "2025-09-30"),
    ("2025-08-01", "2025-08-31"),
    ("2025-07-01", "2025-07-31"),
    ("2025-06-01", "2025-06-30"),
    ("2025-05-01", "2025-05-31"),
    ("2025-04-01", "2025-04-30"),
    ("2025-03-01", "2025-03-31"),
    ("2025-02-01", "2025-02-28"),
]

METHOD_HINTS = re.compile(
    r"\b(model|algorithm|pipeline|classifier|network|regression|clustering|"
    r"simulation|inference|prediction|quantification|segmentation|"
    r"reconstruction|estimation|benchmark|dataset|scoring|mapping|method|"
    r"framework|tool|analysis)\w*\b", re.IGNORECASE)
METRIC_HINTS = re.compile(
    r"\b(accuracy|precision|recall|F1|AUROC|AUC|RMSE|correlation|score|"
    r"error|perplexity|likelihood|sensitivity|specificity|concordance|"
    r"performance|outperform|evaluat)\w*\b", re.IGNORECASE)

# Skip papers already taken by wave1/wave2 (same DOI space).
SKIP_DOIS: set[str] = set()
for _old in ("out/corpus_biorxiv_2026-09.json", "out/corpus_wave2.json", "out/corpus_wave3.json"):
    _p = REPO / _old
    if _p.is_file():
        for _r in json.loads(_p.read_text(encoding="utf-8"))["records"]:
            if _r.get("doi"):
                SKIP_DOIS.add(_r["doi"])


def looks_computational(abstract: str) -> bool:
    if len(abstract) < 500:
        return False
    return bool(METHOD_HINTS.search(abstract)) and bool(METRIC_HINTS.search(abstract))


def _get(url: str, timeout: float = 30.0) -> dict:
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler(urllib.request.getproxies()))
    with opener.open(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def fetch_page(server: str, start: str, end: str, cursor: int) -> list[dict]:
    """One 30-record page. The API's `cursor` is a plain integer offset
    (VERIFIED: offsets 0/30/60 return disjoint sets), so pages within a window
    can be fetched concurrently."""
    url = f"{API}/{server}/{start}/{end}/{cursor}"
    try:
        data = _get(url)
    except Exception:  # noqa: BLE001
        return []
    return data.get("collection") or []


def to_record(hit: dict, server: str) -> dict | None:
    doi = (hit.get("doi") or "").strip()
    title = (hit.get("title") or "").strip()
    abstract = (hit.get("abstract") or "").strip()
    lic = LICENSE_MAP.get((hit.get("license") or "").strip().lower(), "")
    if not doi or not title or not abstract or lic not in OPEN:
        return None
    return {
        "paper_id": doi,
        "title": title,
        "abstract": abstract,
        "doi": doi,
        "pmcid": "",
        "pub_year": int((hit.get("date") or "2026")[:4]) if (hit.get("date") or "")[:4].isdigit() else 2026,
        "published": hit.get("published", "NA"),
        "source_db": f"{server}-api",
        "open_access": True,
        "license": lic,
        "has_code_link": False,
        "has_data_link": False,
        "subjects": [hit.get("category", "")] if hit.get("category") else [],
        "venue": f"{'bioRxiv' if server == 'biorxiv' else 'medRxiv'} preprint",
        "url": f"https://doi.org/{doi}",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=110)
    ap.add_argument("--pages-per-window", type=int, default=40,
                    help="30-record pages per (server, month) = up to 1200 records")
    ap.add_argument("--conn", type=int, default=16)
    args = ap.parse_args()

    # Interleave servers so a full target is reached even from one server.
    jobs = [(s, a, b) for (a, b) in WINDOWS for s in ("biorxiv", "medrxiv")]
    pool = cf.ThreadPoolExecutor(max_workers=args.conn)
    pages = [(s, a, b, i * 30)
             for (s, a, b) in jobs
             for i in range(args.pages_per_window)]
    fetched = list(pool.map(lambda p: (p, fetch_page(*p)), pages))

    # Re-assemble per window, preserving window order and page order.
    window_hits: dict[tuple[str, str, str], list[dict]] = {}
    for (s, a, b, _cur), hits in fetched:
        window_hits.setdefault((s, a, b), []).extend(hits)

    seen: set[str] = set(SKIP_DOIS)
    records: list[dict] = []
    for (server, a, b) in jobs:
        hits = window_hits.get((server, a, b), [])
        got = 0
        for h in hits:
            if len(records) >= args.target:
                break
            doi = (h.get("doi") or "").strip()
            if not doi or doi in seen:
                continue
            if not looks_computational((h.get("abstract") or "")):
                continue
            rec = to_record(h, server)
            if rec is None:
                continue
            seen.add(doi)
            records.append(rec)
            got += 1
        print(f"[{server} {a}..{b}] hits={len(hits)} +{got} (total {len(records)})", flush=True)
        if len(records) >= args.target:
            break

    pool.shutdown(wait=False)
    OUT.write_text(json.dumps({"records": records}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"wrote {len(records)} records -> {OUT}")
    return 0 if len(records) >= 100 else 1


if __name__ == "__main__":
    raise SystemExit(main())
