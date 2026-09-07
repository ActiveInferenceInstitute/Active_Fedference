"""Saved scientific report mappings reproduce exact canonical and slide assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from figures import (
    generate_contamination_gallery,
    generate_robustness_onset,
    generate_robustness_sweep,
)

_ROOT = Path(__file__).resolve().parents[2]


def _asset_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }


@pytest.mark.publication
@pytest.mark.parametrize("name", ("contamination_gallery", "robustness_onset", "robustness_sweep"))
def test_report_mapping_order_does_not_change_figure_or_slide_bytes(
    tmp_path: Path, name: str,
) -> None:
    report = json.loads((_ROOT / "output/reports" / f"{name}.json").read_text())
    field = "accuracy_by_method_and_rate" if name == "robustness_sweep" else "by_kind"
    original = report[field]
    reordered = dict(reversed(tuple(original.items())))
    assert original == reordered
    before = json.dumps(report, sort_keys=True)
    roots = (tmp_path / "saved-order", tmp_path / "reordered")
    for root, values in zip(roots, (original, reordered)):
        if name == "contamination_gallery":
            generate_contamination_gallery(values, project_root=root)
        elif name == "robustness_onset":
            generate_robustness_onset(values, project_root=root)
        else:
            generate_robustness_sweep(
                values,
                report["rates"],
                accuracy_threshold=report["accuracy_threshold"],
                rate_summary=report["per_rate_summary"],
                server_robustness_by_label=report["server_robustness_by_label"],
                project_root=root,
            )
    first, second = map(_asset_hashes, roots)
    assert first and first.keys() == second.keys()
    assert not [name for name in first if first[name] != second[name]]
    assert json.dumps(report, sort_keys=True) == before
