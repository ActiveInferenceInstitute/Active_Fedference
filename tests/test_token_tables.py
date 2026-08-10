"""Token-table arity lint: generated `*_TABLE_ROWS` tokens match their headers.

Several manuscript tables are filled by a single ``{{*_TABLE_ROWS}}`` token whose
value the generator builds as pipe-delimited markdown rows. If the generator's
column count drifts from the header the section declares, the rendered table
silently misaligns. This gate runs the analysis pipeline, hydrates the variables,
and for every section that embeds a ``{{*_TABLE_ROWS}}`` token under a markdown
table header, asserts every generated row has exactly the header's column count.
No mocks: it uses a real pipeline run + the live generator output.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from analysis.workflow import run_analysis_pipeline
from manuscript_variables import generate_variables

_ROOT = Path(__file__).resolve().parent.parent
_MANUSCRIPT = _ROOT / "manuscript"
_TOKEN_ROWS = re.compile(r"\{\{([A-Z0-9_]+_TABLE_ROWS)\}\}")


def _pipe_cols(line: str) -> int:
    """Column count of a markdown table row ``| a | b | c |`` (interior cells)."""
    stripped = line.strip()
    # a well-formed row starts and ends with a pipe; interior count = pipes - 1
    return stripped.count("|") - 1


@pytest.fixture(scope="module")
def variables(tmp_path_factory) -> dict[str, str]:
    root = tmp_path_factory.mktemp("proj")
    # symlink the real manuscript/config so the generator has a project to read,
    # then run the pipeline there to produce reports the tokens consume.
    (root / "manuscript").mkdir()
    (root / "manuscript" / "config.yaml").write_text(
        (_MANUSCRIPT / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    # Keep this real end-to-end fixture bounded. Publication remains the
    # default for the release entry point; the table-arity oracle only needs
    # the same pipeline's schema, not publication-scale sampling.
    run_analysis_pipeline(project_root=root, profile="smoke")
    return generate_variables(root)


def test_every_table_row_token_matches_its_header(variables) -> None:
    offenders: list[str] = []
    sections = sorted(_MANUSCRIPT.glob("[0-9]*.md")) + sorted(_MANUSCRIPT.glob("S[0-9]*.md"))
    for section in sections:
        lines = section.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            m = _TOKEN_ROWS.search(line)
            if not m:
                continue
            token = m.group(1)
            # find the nearest preceding markdown header row (| ... |) then its
            # separator (|---|---|); the header's column count is the contract.
            header_cols = None
            for j in range(i - 1, max(-1, i - 6), -1):
                if set(lines[j].strip()) <= set("|-: "):
                    continue  # separator row
                if lines[j].strip().startswith("|"):
                    header_cols = _pipe_cols(lines[j])
                    break
            value = variables.get(token, "")
            if not value:
                continue  # empty token (e.g. KLD-only family) — nothing to check
            for row in value.splitlines():
                if not row.strip():
                    continue
                cols = _pipe_cols(row)
                if header_cols is not None and cols != header_cols:
                    offenders.append(
                        f"{section.name}: {token} row has {cols} cols, header has {header_cols}: {row[:60]}"
                    )
    assert not offenders, "table token / header arity mismatch:\n  " + "\n  ".join(offenders)
