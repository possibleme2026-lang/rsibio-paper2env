"""M3: Curator — deterministic screening rules + Europe PMC client (offline)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # tests/
sys.path.insert(0, str(ROOT.parent / "src"))

from bioforge_harbor.models.records import PaperRecord
from bioforge_harbor.pipeline.stages.curator import (
    CuratorConfig, curate, score, boundary_papers, refine_with_backend,
)
from bioforge_harbor.retrieval.europepmc import EuropePmcClient, RetrievalError

FIX = ROOT / "fixtures"


def load_corpus() -> list[PaperRecord]:
    data = json.loads((FIX / "corpus" / "mini_corpus.json").read_text(encoding="utf-8"))
    return [PaperRecord.from_dict(d) for d in data]


class TestCuratorRules(unittest.TestCase):
    def setUp(self):
        self.recs = {r.paper_id: r for r in load_corpus()}
        self.cfg = CuratorConfig(select_n=100)

    def _reason(self, pid: str) -> list[str]:
        s, bucket, reasons = score(self.recs[pid], self.cfg)
        return reasons if s < 0 else []

    def test_computational_oa_selected(self):
        for pid in ["epmc-PMC100", "epmc-PMC101", "epmc-PMC103",
                    "epmc-PMC105", "epmc-PMC106", "epmc-PMC110"]:
            self.assertEqual(self._reason(pid), [], f"{pid} should pass, got {self._reason(pid)}")

    def test_not_open_access_rejected(self):
        self.assertIn("not_open_access", self._reason("epmc-PMC104"))

    def test_year_out_of_range_rejected(self):
        self.assertIn("year_out_of_range:2025", self._reason("epmc-PMC107"))

    def test_abstract_too_short_rejected(self):
        self.assertIn("abstract_too_short", self._reason("epmc-PMC109"))

    def test_wetlab_is_boundary_not_hard_reject(self):
        # PMC102 (wet-lab) and PMC111 (qualitative survey) fail ONLY the
        # computational rule -> they are boundary cases, not certain rejects.
        self.assertEqual(self._reason("epmc-PMC102"), ["not_computational"])
        self.assertEqual(self._reason("epmc-PMC111"), ["not_computational"])
        bps = {r.paper_id for r in boundary_papers(list(self.recs.values()), self.cfg)}
        self.assertEqual(bps, {"epmc-PMC102", "epmc-PMC111"})

    def test_nd_license_still_screenable(self):
        # CC BY-NC-ND is open-access-readable; implementing an algorithm is
        # not a derivative of the paper text, so it is not a hard reject.
        self.assertEqual(self._reason("epmc-PMC108"), [])


class TestCuratorSelection(unittest.TestCase):
    def test_select_n_caps_selection(self):
        recs = load_corpus()
        rep = curate(recs, CuratorConfig(select_n=3))
        self.assertEqual(rep.selected, 3)
        self.assertLessEqual(len(rep.selected_ids), 3)

    def test_deterministic_ordering(self):
        recs = load_corpus()
        a = curate(recs, CuratorConfig(select_n=100)).selected_ids
        b = curate(recs, CuratorConfig(select_n=100)).selected_ids
        self.assertEqual(a, b)

    def test_ranking_prefers_code_and_metric(self):
        recs = load_corpus()
        ids = curate(recs, CuratorConfig(select_n=100)).selected_ids
        # PMC101 (code+data+metric) must outrank a bare-pass paper like PMC108
        self.assertLess(ids.index("epmc-PMC101"), ids.index("epmc-PMC108"))

    def test_bucket_cap(self):
        recs = load_corpus()
        cfg = CuratorConfig(select_n=100, buckets={"single-cell-rnaseq": 1})
        rep = curate(recs, cfg)
        # only one single-cell paper allowed through
        sc = [pid for pid in rep.selected_ids if pid == "epmc-PMC100"]
        self.assertEqual(len(sc), 1)
        self.assertTrue(any(r["reason"].startswith("bucket_cap") for r in rep.rejected)
                        or rep.bucket_counts.get("single-cell-rnaseq", 0) <= 1)

    def test_report_shape(self):
        rep = curate(load_corpus(), CuratorConfig(select_n=100))
        self.assertEqual(rep.stage, "curate")
        self.assertEqual(rep.screened, 12)
        self.assertEqual(rep.selected, len(rep.selected_ids))
        self.assertTrue(rep.ok)


class TestRefineWithBackend(unittest.TestCase):
    def test_backend_only_sees_boundary_papers(self):
        from bioforge_harbor.backends import MockBackend
        calls: list[str] = []

        class Spy(MockBackend):
            def ask_json(self, tag, prompt, *, schema_name=None, timeout_sec=600.0):
                calls.append(prompt)
                return {"computable": "single-cell" in prompt.lower()}

        recs = load_corpus()
        verdicts = refine_with_backend(recs, Spy({}))
        # exactly the two boundary papers were queried
        self.assertEqual(set(verdicts), {"epmc-PMC102", "epmc-PMC111"})
        self.assertEqual(len(calls), 2)


class TestEuropePmcClient(unittest.TestCase):
    def _resp(self):
        return json.loads((FIX / "europepmc" / "search_response.json").read_text(encoding="utf-8"))

    def test_search_parses_records(self):
        payload = self._resp()

        def fetcher(url):
            return json.dumps(payload).encode("utf-8"), "application/json"

        c = EuropePmcClient(fetcher=fetcher)
        resp = c.search("single-cell")
        hits = resp["resultList"]["result"]
        recs = [PaperRecord.from_europepmc(h) for h in hits]
        self.assertEqual(recs[0].paper_id, "epmc-PMC1234567")
        self.assertTrue(recs[0].open_access)

    def test_html_error_page_rejected(self):
        html = (FIX / "europepmc" / "error_page.html").read_bytes()

        def fetcher(url):
            return html, "text/html"

        c = EuropePmcClient(fetcher=fetcher, max_retries=1, backoff_sec=0)
        with self.assertRaises(RetrievalError):
            c.search("x")

    def test_cache_hit_is_offline(self):
        import tempfile
        d = Path(tempfile.mkdtemp())
        payload = self._resp()

        def fetcher(url):
            return json.dumps(payload).encode("utf-8"), "application/json"

        c = EuropePmcClient(fetcher=fetcher, cache_dir=d)
        c.search("single-cell")
        # second call: offline client with a populated cache must not fetch
        offline = EuropePmcClient(fetcher=None, cache_dir=d, offline=True)
        # reconstruct same URL by using the same query
        resp = offline.search("single-cell")
        self.assertIn("resultList", resp)

    def test_search_all_pages(self):
        def make(cursor_val, hits):
            return {"resultList": {"result": hits}, "nextCursor": cursor_val}

        page1 = make("c2", [{"pmcid": "PMC1", "title": "a", "doi": ""}])
        page2 = make("na", [{"pmcid": "PMC2", "title": "b", "doi": ""}])
        seq = [page1, page2]

        def fetcher(url):
            return json.dumps(seq.pop(0)).encode("utf-8"), "application/json"

        c = EuropePmcClient(fetcher=fetcher)
        allhits = c.search_all("x", page_size=1)
        self.assertEqual([h["pmcid"] for h in allhits], ["PMC1", "PMC2"])


if __name__ == "__main__":
    unittest.main()
