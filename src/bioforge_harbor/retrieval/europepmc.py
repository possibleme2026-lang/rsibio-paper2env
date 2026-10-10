"""Europe PMC REST client.

Design points that matter more than the API details:

* **A pluggable ``fetcher``.** The default uses ``urllib`` but every network
  touch goes through one callable. Tests inject a fixture-backed fetcher, so
  the whole retrieval layer is exercised offline; an operator can inject a
  proxy-aware or cached fetcher without touching this module.
* **Reject HTML served with 200.** Europe PMC (and ChEMBL) sometimes return an
  HTML error page under a 200 status. We refuse anything that does not parse
  as JSON and is not obviously JSON.
* **Offline-first cache.** A response cache directory lets a re-run be fully
  deterministic and network-free, and records the digest of what was fetched.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
USER_AGENT = "rsibio-paper2env/0.1 (research; contact: RisingSI)"

# Fields we always request. Kept explicit so a silent upstream schema change
# shows up as a missing key rather than a subtly-empty record.
RESULT_FIELDS = ",".join([
    "id", "pmcid", "doi", "title", "abstractText", "pubYear", "pubDate",
    "journalTitle", "isOpenAccess", "licence", "inEPMC", "hasSupplFiles",
    "subjectTerms", "source",
])


class RetrievalError(RuntimeError):
    pass


Fetcher = Callable[[str], tuple[bytes, str]]  # url -> (body, content_type)


def _urllib_fetcher(url: str, *, timeout: float = 30.0) -> tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            ctype = resp.headers.get("Content-Type", "")
            return resp.read(), ctype
    except urllib.error.HTTPError as e:  # pragma: no cover - network path
        raise RetrievalError(f"HTTP {e.code} for {url}") from e
    except urllib.error.URLError as e:  # pragma: no cover - network path
        raise RetrievalError(f"network error for {url}: {e}") from e


@dataclass
class EuropePmcClient:
    fetcher: Fetcher | None = None
    cache_dir: Path | None = None
    max_retries: int = 3
    backoff_sec: float = 1.0
    offline: bool = False

    def __post_init__(self):
        self._fetcher: Fetcher = self.fetcher or _urllib_fetcher
        if self.cache_dir:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- cache ------------------------------------------------------------
    def _cache_path(self, url: str) -> Path | None:
        if not self.cache_dir:
            return None
        from ..util.io import sha256_text
        return self.cache_dir / (sha256_text(url)[:32] + ".json")

    def _get(self, url: str) -> dict:
        cp = self._cache_path(url)
        if cp and cp.is_file():
            return json.loads(cp.read_text(encoding="utf-8"))
        if self.offline:
            raise RetrievalError(f"offline and no cache for {url}")

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                body, ctype = self._fetcher(url)
                return self._parse(body, ctype, url, cp)
            except RetrievalError as e:
                last_err = e
                time.sleep(self.backoff_sec * (attempt + 1))
        raise RetrievalError(f"failed after {self.max_retries} tries: {last_err}")

    def _parse(self, body: bytes, ctype: str, url: str, cp: Path | None) -> dict:
        text = body.decode("utf-8", errors="replace")
        looks_html = text.lstrip()[:15].lower().startswith(("<!doctype", "<html"))
        if "html" in (ctype or "").lower() or looks_html:
            # 200 + HTML error page: refuse rather than return garbage.
            raise RetrievalError(f"expected JSON, got HTML for {url}")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise RetrievalError(f"invalid JSON for {url}: {e}") from e
        if cp:
            cp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data

    # -- public API -------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        result_type: str = "core",
        page: int = 1,
        page_size: int = 25,
        cursor: str | None = None,
    ) -> dict:
        """One page of a search. Returns the raw response dict."""
        params = {
            "query": query,
            "format": "json",
            "resultType": result_type,
            "pageSize": str(min(page_size, 1000)),
            "page": str(page),
        }
        if cursor:
            params["cursor"] = cursor
        url = f"{BASE}/search?{urllib.parse.urlencode(params)}"
        return self._get(url)

    def search_all(self, query: str, *, max_records: int = 1000, page_size: int = 200) -> list[dict]:
        """Iterate cursor-based paging up to ``max_records`` hits."""
        out: list[dict] = []
        cursor = "*"
        while len(out) < max_records and cursor:
            resp = self.search(query, page_size=page_size, cursor=cursor)
            hits = resp.get("resultList", {}).get("result", []) or []
            out.extend(hits)
            next_cursor = (resp.get("nextCursor") or "").strip()
            if not hits or next_cursor == "na" or next_cursor == cursor:
                break
            cursor = next_cursor
        return out[:max_records]

    def fetch_article(self, id_type: str, id: str) -> dict:
        """Full metadata for one article (``FULL_JSON`` format)."""
        url = f"{BASE}/{id_type}/{id}/textMinedTerms?format=json"
        return self._get(url)

    def supplementary_files(self, pmcid: str) -> list[dict]:
        """List supplementary file entries for a PMC id (``PMC1234567``)."""
        url = f"{BASE}/{pmcid}/supplementaryFiles?format=json"
        data = self._get(url)
        # The endpoint returns an XML-derived dict; be defensive about shape.
        files = data.get("fileList", {}).get("file", []) if isinstance(data, dict) else []
        if isinstance(files, dict):
            files = [files]
        return files if isinstance(files, list) else []
