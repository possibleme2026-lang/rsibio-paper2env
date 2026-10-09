"""PaperRecord: the normalized corpus entry every stage consumes.

A record is deliberately a *plain dict with validators* rather than a heavy
ORM-ish object: it round-trips through JSON on disk (corpus files, run state)
and must stay diffable. ``from_europepmc`` maps the Europe PMC REST field names
we rely on; unknown fields are ignored, missing required fields raise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any

# Europe PMC license codes that allow (re)use of full text / metadata for our
# screening pipeline. Anything not listed is treated as "restricted".
OPEN_ACCESS_LICENSES = {
    "CC BY",
    "CC BY-SA",
    "CC BY-NC",
    "CC BY-NC-SA",
    "CC BY-ND",
    "CC BY-NC-ND",
    "CC0",
    "PUBLIC DOMAIN",
}

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_len: int = 48) -> str:
    s = _SLUG_RE.sub("-", (text or "").lower()).strip("-")
    if not s:
        s = "untitled"
    return s[:max_len].rstrip("-") or "untitled"


@dataclass
class PaperRecord:
    """One screened publication. ``paper_id`` is stable across runs."""

    paper_id: str                     # e.g. "epmc-PMC1234567" or "biorxiv-10.1101/xxxx"
    title: str
    abstract: str = ""
    doi: str = ""
    pmcid: str = ""                   # "PMC..." if present
    pub_year: int = 0                 # 0 = unknown
    published: str = ""               # raw date string from source
    source_db: str = "europepmc"      # europepmc | pmc | biorxiv
    open_access: bool = False
    license: str = ""
    has_code_link: bool = False
    has_data_link: bool = False
    subjects: list[str] = field(default_factory=list)
    venue: str = ""
    url: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    REQUIRED = ("paper_id", "title")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PaperRecord":
        missing = [k for k in cls.REQUIRED if not d.get(k)]
        if missing:
            raise ValueError(f"PaperRecord missing required fields: {missing}")
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        extra = {k: v for k, v in d.items() if k not in known}
        kwargs = {k: v for k, v in d.items() if k in known and k != "extra"}
        if d.get("extra"):
            extra.update(d["extra"])
        kwargs.setdefault("extra", extra)
        return cls(**kwargs)

    @property
    def year_or_zero(self) -> int:
        if self.pub_year:
            return self.pub_year
        m = re.search(r"(19|20)\d{2}", self.published or "")
        return int(m.group(0)) if m else 0

    @classmethod
    def from_europepmc(cls, hit: dict[str, Any]) -> "PaperRecord":
        """Map one Europe PMC ``resultList.result[]`` entry to a record.

        Field names follow the Europe PMC REST search API (webservice=rest).
        """
        pmcid = (hit.get("pmcid") or "").strip()
        doi = (hit.get("doi") or "").strip()
        title = (hit.get("title") or "").strip()
        pub_year = 0
        py = hit.get("pubYear")
        if py is not None:
            try:
                pub_year = int(py)
            except (TypeError, ValueError):
                pub_year = 0
        paper_id = f"epmc-{pmcid}" if pmcid else (f"doi-{doi}" if doi else "")
        if not paper_id:
            raise ValueError("Europe PMC hit has neither pmcid nor doi")
        license_ = (hit.get("licence") or "").strip().upper()
        subjects = hit.get("subjectTerms") or []
        if isinstance(subjects, dict):
            subjects = [t.get("text", "") for t in subjects.get("subjectTerm", []) if isinstance(t, dict)]
        return cls(
            paper_id=paper_id,
            title=title,
            abstract=(hit.get("abstractText") or ""),
            doi=doi,
            pmcid=pmcid,
            pub_year=pub_year,
            published=(hit.get("pubDate") or ""),
            source_db="europepmc",
            open_access=str(hit.get("isOpenAccess", "")).upper() == "Y",
            license=license_,
            # Code/data availability is not reliably present in search hits;
            # downstream stages may upgrade these flags from full-text mining.
            has_code_link=bool(hit.get("inEPMC", "") == "Y" and hit.get("hasSupplFiles")),
            subjects=[str(s) for s in subjects if s],
            venue=(hit.get("journalTitle") or ""),
            url=(f"https://europepmc.org/article/PMC/{pmcid[3:]}" if pmcid.startswith("PMC")
                 else (f"https://doi.org/{doi}" if doi else "")),
        )
