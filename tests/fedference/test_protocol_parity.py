"""Protocol matrices enforce exact-replication labeling boundaries."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from fedference.protocol_parity import (
    ParityRow,
    ProtocolParityMatrix,
    fedgvi_bnn_parity_matrix,
    friston_protocol_parity_matrices,
    run_friston_protocol_audit,
)


def test_fedgvi_matrix_records_site_protocol_and_portable_deviations() -> None:
    matrix = fedgvi_bnn_parity_matrix()
    assert matrix.exact is False
    assert matrix.claim_label == "source-constrained implementation"
    server = next(row for row in matrix.rows if row.field == "server update")
    assert server.status == "matched"
    assert "site factor" in server.project_value
    seeds = next(row for row in matrix.rows if row.field == "seed indexing")
    assert seeds.status == "matched"
    assert "[676, 93, 215, 318, 242]" in seeds.source_value
    assert {
        "client split",
        "loss and client divergence",
        "local stopping",
        "posterior predictive samples",
        "ELBO samples",
    } <= {row.field for row in matrix.rows}
    assert any(row.status == "deviation" for row in matrix.rows)
    json.dumps(matrix.as_dict())


def test_friston_matrices_cannot_claim_exact_while_source_fields_are_unknown() -> None:
    matrices = friston_protocol_parity_matrices()
    assert len(matrices) == 4
    assert all(matrix.claim_label == "paper-constrained reconstruction" for matrix in matrices)
    assert all(any(row.status == "unresolved" for row in matrix.rows) for matrix in matrices)


def test_friston_audit_keeps_analogue_relabeling_negative_control() -> None:
    report = run_friston_protocol_audit()
    assert report["status"] == "paper-constrained reconstruction"
    assert report["negative_control"]["passed"] is True
    assert len(report["protocols"]) == 4


def test_only_all_matched_rows_emit_exact_replication() -> None:
    row = ParityRow("field", "source", "project", "matched", "test")
    exact = ProtocolParityMatrix("protocol", "source", "target", (row,))
    assert exact.exact
    assert exact.claim_label == "exact replication"
    with pytest.raises(ValueError, match="unique"):
        replace(exact, rows=(row, row))
    with pytest.raises(ValueError, match="field"):
        replace(row, field=3)
    with pytest.raises(ValueError, match="ParityRow"):
        replace(exact, rows=("not-a-row",))


def test_parity_matrix_owns_an_immutable_row_sequence() -> None:
    row = ParityRow("field", "source", "project", "matched", "test")
    source_rows = [row]
    matrix = ProtocolParityMatrix("protocol", "source", "target", source_rows)
    source_rows.clear()
    assert matrix.rows == (row,)
    assert isinstance(matrix.rows, tuple)


def test_parity_contract_rejects_invalid_status_and_schema() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ParityRow("", "source", "project", "matched", "test")
    with pytest.raises(ValueError, match="recognized"):
        ParityRow("field", "source", "project", "invalid", "test")  # type: ignore[arg-type]
    row = ParityRow("field", "source", "project", "not-applicable", "test")
    assert ProtocolParityMatrix("protocol", "source", "target", (row,)).exact is False
    with pytest.raises(ValueError, match="rows"):
        ProtocolParityMatrix("protocol", "source", "target", ())
    with pytest.raises(ValueError, match="schema"):
        ProtocolParityMatrix("protocol", "source", "target", (row,), schema_version="2.0")
