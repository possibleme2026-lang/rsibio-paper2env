"""M7: the guards guard something — each one fails on a real violation.

A guard that only ever passes is decoration. Every guard here is run twice:
once against this repository (must be clean) and once against a deliberately
broken fixture (must fail). The negative cases are the actual test.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # repo root
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import check_claims
import check_lf
import check_readme_i18n

CLAIMS = ROOT / "claims.json"
PAPER = ROOT / "paper" / "main.tex"
LEDGER = ROOT / "paper" / "sections" / "claims_ledger.tex"


# --------------------------------------------------------------------------
# claims
# --------------------------------------------------------------------------
def _claim(cid: str = "X1", **over) -> dict:
    c = {"id": cid, "statement": "A plain structural statement with no assertion.",
         "kind": "structural", "status": "measured", "metric": None, "control": None,
         "evidence": ["pyproject.toml"]}
    c.update(over)
    return c


class TestClaimsGuard(unittest.TestCase):
    def test_repo_claims_are_clean(self):
        self.assertEqual(check_claims.main(["--claims", str(CLAIMS), "--root", str(ROOT)]), 0)

    def test_repo_claims_have_planned_and_measured(self):
        data = json.loads(CLAIMS.read_text(encoding="utf-8"))
        statuses = {c["status"] for c in data["claims"]}
        # the framework-only release must not claim a single measured number
        self.assertIn("measured", statuses)
        self.assertIn("planned", statuses)

    def test_measured_without_evidence_fails(self):
        problems = check_claims.check({"claims": [_claim(evidence=[])]}, root=ROOT)
        self.assertTrue(any("no evidence" in p for p in problems), problems)

    def test_measured_with_missing_evidence_file_fails(self):
        problems = check_claims.check(
            {"claims": [_claim(evidence=["does/not/exist.py"])]}, root=ROOT)
        self.assertTrue(any("not found on disk" in p for p in problems), problems)

    def test_comparative_without_control_fails(self):
        problems = check_claims.check(
            {"claims": [_claim(kind="comparative", control=None,
                               metric="control_contrast")]}, root=ROOT)
        self.assertTrue(any("control group" in p for p in problems), problems)

    def test_rate_without_metric_fails(self):
        problems = check_claims.check(
            {"claims": [_claim(kind="rate", metric=None,
                               control="a control arm")]}, root=ROOT)
        self.assertTrue(any("must name its metric" in p for p in problems), problems)

    def test_control_group_first_catches_an_uncontrolled_assertion(self):
        # filed as 'structural' but the wording makes a comparative claim:
        # exactly the drift this rule exists to catch
        problems = check_claims.check(
            {"claims": [_claim(statement="Auto-built environments outperform random ones.",
                               kind="structural", control=None)]}, root=ROOT)
        self.assertTrue(any("declares no control" in p for p in problems), problems)

    def test_duplicate_ids_fail(self):
        problems = check_claims.check({"claims": [_claim("D1"), _claim("D1")]}, root=ROOT)
        self.assertTrue(any("duplicate" in p for p in problems), problems)

    def test_bad_status_fails(self):
        problems = check_claims.check({"claims": [_claim(status="totally measured")]},
                                      root=ROOT)
        self.assertTrue(any("status" in p for p in problems), problems)

    def test_paper_reference_must_be_registered(self):
        tmp = Path(tempfile.mkdtemp())
        paper = tmp / "main.tex"
        paper.write_text("As shown in \\cref{C1} and P9 the numbers hold.", encoding="utf-8")
        data = json.loads(CLAIMS.read_text(encoding="utf-8"))
        # C1 and P9 are registered -> clean
        self.assertEqual(check_claims.check(data, root=ROOT, paper=paper), [])
        paper.write_text("See P99 for the headline number.", encoding="utf-8")
        problems = check_claims.check(data, root=ROOT, paper=paper)
        self.assertTrue(any("P99" in p for p in problems), problems)


# --------------------------------------------------------------------------
# the manuscript itself
# --------------------------------------------------------------------------
@unittest.skipUnless(PAPER.is_file(), "paper/main.tex not present")
class TestManuscript(unittest.TestCase):
    def test_every_claim_id_in_the_paper_is_registered(self):
        data = json.loads(CLAIMS.read_text(encoding="utf-8"))
        self.assertEqual(check_claims.check(data, root=ROOT, paper=PAPER), [])

    def test_claims_ledger_is_in_sync_with_the_registry(self):
        # Appendix B is generated; if someone edits claims.json without
        # regenerating it, the appendix is silently wrong
        self.assertTrue(LEDGER.is_file(), "claims_ledger.tex missing")
        import gen_claims_tex
        text = gen_claims_tex.render(json.loads(CLAIMS.read_text(encoding="utf-8")))
        self.assertEqual(LEDGER.read_text(encoding="utf-8"), text)

    def test_paper_declares_no_measured_number_without_evidence(self):
        claims = json.loads(CLAIMS.read_text(encoding="utf-8"))["claims"]
        for c in claims:
            if c["status"] == "measured":
                self.assertTrue(c["evidence"], f"{c['id']}: measured with no evidence")


# --------------------------------------------------------------------------
# line endings
# --------------------------------------------------------------------------
class TestLfGuard(unittest.TestCase):
    def test_repo_is_clean(self):
        self.assertEqual(check_lf.main(["--root", str(ROOT)]), 0)

    def test_crlf_is_caught(self):
        tmp = Path(tempfile.mkdtemp())
        bad = tmp / "bad.sh"
        bad.write_bytes(b"#!/bin/bash\r\nset -e\r\n")
        problems = check_lf.check(tmp)
        self.assertTrue(any("CRLF" in p for p in problems), problems)

    def test_bom_is_caught(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "bom.py").write_bytes(b"\xef\xbb\xbfprint('hi')\n")
        problems = check_lf.check(tmp)
        self.assertTrue(any("BOM" in p for p in problems), problems)

    def test_lf_file_passes(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "ok.py").write_bytes(b"print('hi')\n")
        self.assertEqual(check_lf.check(tmp), [])


# --------------------------------------------------------------------------
# README i18n
# --------------------------------------------------------------------------
HEAD = "# Title\n\n## A\n\ntext\n\n## B\n\ntext\n"


class TestReadmeI18nGuard(unittest.TestCase):
    def test_repo_readmes_are_in_parity(self):
        self.assertEqual(check_readme_i18n.main(["--root", str(ROOT)]), 0)

    def test_identical_skeletons_pass(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "README.md").write_text(HEAD, encoding="utf-8")
        (tmp / "README.zh-CN.md").write_text("# 标题\n\n## 甲\n\n正文\n\n## 乙\n\n正文\n",
                                             encoding="utf-8")
        self.assertEqual(check_readme_i18n.main(["--root", str(tmp)]), 0)

    def test_an_extra_section_in_one_language_fails(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "README.md").write_text(HEAD + "\n## C\n\ntext\n", encoding="utf-8")
        (tmp / "README.zh-CN.md").write_text(HEAD, encoding="utf-8")
        self.assertEqual(check_readme_i18n.main(["--root", str(tmp)]), 1)

    def test_a_link_only_in_one_language_fails(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "README.md").write_text(HEAD + "\nsee https://example.org/only-en\n",
                                       encoding="utf-8")
        (tmp / "README.zh-CN.md").write_text(HEAD, encoding="utf-8")
        self.assertEqual(check_readme_i18n.main(["--root", str(tmp)]), 1)

    def test_a_missing_translation_fails(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "README.md").write_text(HEAD, encoding="utf-8")
        self.assertEqual(check_readme_i18n.main(["--root", str(tmp)]), 1)


if __name__ == "__main__":
    unittest.main()
