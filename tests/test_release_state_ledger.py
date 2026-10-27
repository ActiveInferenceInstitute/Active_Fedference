"""Offline release-ledger checks; live hosting is verified in dated HTTP evidence.

These tests exercise the production immutable-PDF validator using real files.
They do not fetch network data or certify current GitHub/Zenodo freshness.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, NoReturn

import pytest

from publication.clean_checkout import (
    HISTORICAL_RELEASE_PDF_LEDGER_PATH,
    IMMUTABLE_RELEASE_PDFS,
    historical_release_pdf_findings,
)

ROOT = Path(__file__).resolve().parents[1]


def _assert_never(value: NoReturn) -> NoReturn:
    raise AssertionError(value)


def test_release_identity_and_downloaded_bytes() -> None:
    """The dated observation agrees with the immutable local release bytes."""
    # Given: the durable observed-data block and immutable source ledger.
    page = (ROOT / "docs/research/release-state-20261027.md").read_text()
    observed = json.loads(page.split("```json\n", 1)[1].split("\n```", 1)[0])
    ledger = json.loads((ROOT / HISTORICAL_RELEASE_PDF_LEDGER_PATH).read_text())
    pdf = ROOT / observed["pdf"]["name"]
    # When: the production verifier reads all six PDFs and their receipt.
    findings = historical_release_pdf_findings(ROOT)
    # Then: actual bytes bind the observed release identity without a live claim.
    assert findings == ()
    assert observed["doi"] == f"10.5281/zenodo.{observed['zenodo_record_id']}"
    assert observed["concept_doi"] == f"10.5281/zenodo.{observed['zenodo_concept_id']}"
    assert f"_v{observed['version']}_" in pdf.name
    assert observed["doi"].replace("/", "-") in pdf.name
    assert observed["pdf"]["bytes"] == pdf.stat().st_size
    assert observed["pdf"]["sha256"] == hashlib.sha256(pdf.read_bytes()).hexdigest()
    assert observed["pdf"]["sha256"] == ledger["sha256"][pdf.name]
    assert observed["zenodo_pdf_sha256"] == observed["pdf"]["sha256"]


@pytest.mark.parametrize("corruption", ["asset", "receipt", "malformed", "duplicate"])
def test_asset_digest_or_receipt_mismatch(
    tmp_path: Path, corruption: Literal["asset", "receipt", "malformed", "duplicate"]
) -> None:
    """Corrupt bytes or receipts cannot yield a clean immutable-PDF finding."""
    # Given: real files and an independently assembled receipt.
    digests = {}
    for index, filename in enumerate(IMMUTABLE_RELEASE_PDFS):
        content = f"%PDF-1.7\nfixture {index}\n".encode()
        (tmp_path / filename).write_bytes(content)
        digests[filename] = hashlib.sha256(content).hexdigest()
    receipt = tmp_path / HISTORICAL_RELEASE_PDF_LEDGER_PATH
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"schema_version": "1.0", "sha256": digests}))
    assert historical_release_pdf_findings(tmp_path) == ()
    filename = IMMUTABLE_RELEASE_PDFS[-1]
    match corruption:
        case "asset":
            (tmp_path / filename).write_bytes(b"%PDF-1.7\naltered fixture\n")
        case "receipt":
            digests[filename] = "0" * 64
            receipt.write_text(json.dumps({"schema_version": "1.0", "sha256": digests}))
        case "malformed":
            receipt.write_text('{"schema_version": "1.0", "sha256": []}')
        case "duplicate":
            receipt.write_text('{"schema_version":"1.0","sha256":{},"sha256":{}}')
        case unreachable:
            _assert_never(unreachable)
    before = {p.relative_to(tmp_path).as_posix(): p.read_bytes()
              for p in tmp_path.rglob("*") if p.is_file()}
    # When: the same production verifier sees the tampered snapshot.
    findings = historical_release_pdf_findings(tmp_path)
    # Then: closure fails and the original failure inputs are retained unchanged.
    assert findings
    if corruption in {"asset", "receipt"}:
        assert any("SHA-256 mismatch" in finding and filename in finding for finding in findings)
    else:
        assert any("invalid" in finding or "malformed" in finding for finding in findings)
    assert before == {p.relative_to(tmp_path).as_posix(): p.read_bytes()
                      for p in tmp_path.rglob("*") if p.is_file()}
    print(json.dumps({"case": corruption, "accepted": False, "findings": findings,
                      "inputs_unchanged": True}))
