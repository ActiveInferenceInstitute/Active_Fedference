"""Real rendering, source placement and scientific-claim controls; no mocks."""

from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from matplotlib.text import Annotation, Text
from PIL import Image

from analysis.artifacts import ANALYSIS_FIGURE_FILENAMES, PRESENTATION_FIGURE_STEMS
from analysis.visual_contracts import COMPLEX_FIGURE_GENERATORS
from analysis.workflow import _declared_manuscript_figures, _write_figure_registry, _write_json
from fedference.aggregation import log_linear_pool, robust_aggregate
from figures import FIGURE_METADATA
from figures._common import plt, validate_figure_text
from figures._frontmatter import FRONTMATTER, validate_frontmatter
from figures._presentation import validate_presentation_files

ROOT = Path(__file__).resolve().parents[2]


def _fixture(destination: Path) -> Path:
    manuscript = destination / "manuscript"
    manuscript.mkdir(parents=True)
    shutil.copyfile(ROOT / "manuscript/config.yaml", manuscript / "config.yaml")
    for _, name in FRONTMATTER:
        shutil.copyfile(ROOT / "manuscript" / name, manuscript / name)
    return destination


@pytest.mark.parametrize("stem,filename", FRONTMATTER)
def test_frontmatter_pair_geometry_and_repeatability(tmp_path: Path, stem: str, filename: str) -> None:
    module = importlib.import_module(f"figures.{stem}")
    fig = module.build_figure()
    try:
        validate_figure_text(fig, minimum_font_size=8.5, manuscript_width_fraction=1)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
        assert min(t.get_fontsize() for t in texts) >= 10.5
        for i, text in enumerate(texts):
            box = text.get_window_extent(renderer)
            assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1), text.get_text()
            for other in texts[i + 1 :]:
                assert not box.overlaps(other.get_window_extent(renderer)), (
                    text.get_text(),
                    other.get_text(),
                )
        assert len([a for a in fig.findobj(Annotation) if a.arrow_patch is not None]) >= 3
    finally:
        plt.close(fig)
    generator = getattr(module, f"generate_{stem}")
    first = generator(project_root=tmp_path / "first")
    second = generator(project_root=tmp_path / "second")
    for suffix in (".png", ".pdf"):
        assert first.with_suffix(suffix).read_bytes() == second.with_suffix(suffix).read_bytes()
    with Image.open(first) as image:
        assert image.width > 1200 and image.height > image.width
    assert first.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
    assert validate_presentation_files(first.with_suffix(".slides.json"))
    assert stem + ".png" in ANALYSIS_FIGURE_FILENAMES
    assert stem in PRESENTATION_FIGURE_STEMS
    # Deferred reader surfaces: a plate rejoins COMPLEX_FIGURE_GENERATORS with
    # its structured long description once the pinned producer renders the
    # matching 00a-00c HTML page; see docs/todo/visual-scholarship-integration.md.
    assert stem not in COMPLEX_FIGURE_GENERATORS
    assert "long_description" not in FIGURE_METADATA[stem]
    assert (ROOT / "manuscript" / filename).is_file()


def test_frontmatter_placement_and_registry_inputs() -> None:
    validate_frontmatter(ROOT)
    declarations = _declared_manuscript_figures(ROOT)
    pages = [entry for entry in declarations if entry["filename"].startswith("frontmatter_")]
    assert [p["filename"] for p in pages] == [
        "frontmatter_model.png",
        "frontmatter_robustness.png",
        "frontmatter_execution.png",
    ]
    for page in pages:
        assert "no measured estimand" in page["caption"]
        assert "data-slide-manifest" not in page["caption"]


def test_frontmatter_canonical_registry_boundary(tmp_path: Path) -> None:
    from analysis.figure_exact_values import (
        EXACT_VALUE_REPORT_BINDINGS,
        render_figure_exact_values_markdown,
    )

    root = _fixture(tmp_path)
    paths = {}
    for stem, _ in FRONTMATTER:
        module = importlib.import_module(f"figures.{stem}")
        paths[stem] = getattr(module, f"generate_{stem}")(project_root=root)
    # The shared writer requires table artifacts even for this conceptual-only
    # fixture. Real temporary files carry explicitly non-numerical test rows;
    # no historical report is regenerated or represented as current evidence.
    exact = {
        "schema_version": "1.0",
        "generated_by": "analysis.figure_exact_values.build_figure_exact_values",
        "tables": [
            {
                "identifier": "fig-values:" + generator.replace("_", "-"),
                "generator": generator,
                "source_report": report + ".json",
                "columns": ["fixture"],
                "rows": [{"fixture": "Not consumed by the conceptual figures"}],
                "note": "Non-numerical temporary registry-boundary fixture, not scientific evidence.",
            }
            for generator, report in EXACT_VALUE_REPORT_BINDINGS
        ],
    }
    paths["figure_exact_values"] = _write_json(
        exact, root / "output/figures/figure_exact_values.json", schema="figure_exact_values",
    )
    markdown = root / "output/figures/figure_exact_values.md"
    markdown.write_text(render_figure_exact_values_markdown(exact))
    paths["figure_exact_values_markdown"] = markdown
    registry = _write_figure_registry(root, paths)
    original_registry = registry.read_bytes()
    entries = json.loads(registry.read_text())["figures"]
    assert {entry["generated_by"] for entry in entries} == {stem for stem, _ in FRONTMATTER}
    assert all(entry["presentation_manifest"] for entry in entries)
    assert all("long_description" not in entry for entry in entries)
    missing = paths["frontmatter_model"]
    original = missing.read_bytes()
    try:
        missing.unlink()
        with pytest.raises(FileNotFoundError, match="Declared manuscript figure is missing"):
            _write_figure_registry(root, paths)
    finally:
        missing.write_bytes(original)
    assert _write_figure_registry(root, paths).read_bytes() == original_registry


@pytest.mark.parametrize("defect", ["missing", "ordering", "embed", "caption"])
def test_frontmatter_rejects_real_source_defects(tmp_path: Path, defect: str) -> None:
    root = _fixture(tmp_path)
    source = root / "manuscript" / FRONTMATTER[0][1]
    config = root / "manuscript/config.yaml"
    original_source, original_config = source.read_bytes(), config.read_bytes()
    try:
        if defect == "missing":
            source.unlink()
        elif defect == "ordering":
            data = yaml.safe_load(config.read_text())
            data["paper"]["frontmatter"].reverse()
            config.write_text(yaml.safe_dump(data))
        elif defect == "embed":
            source.write_text(source.read_text().replace("frontmatter_model.png", "absent.png"))
        else:
            source.write_text(source.read_text().replace("Source relation:", "Removed relation:"))
        with pytest.raises((FileNotFoundError, ValueError)):
            validate_frontmatter(root)
    finally:
        source.write_bytes(original_source)
        config.write_bytes(original_config)
    assert source.read_bytes() == original_source and config.read_bytes() == original_config
    validate_frontmatter(root)


def test_frontmatter_optional_fixture_and_required_configuration(tmp_path: Path) -> None:
    validate_frontmatter(tmp_path, required=False)
    (tmp_path / "manuscript").mkdir()
    (tmp_path / "manuscript/config.yaml").write_text("paper: {}\n")
    validate_frontmatter(tmp_path, required=False)
    with pytest.raises(ValueError, match="order"):
        validate_frontmatter(tmp_path)


@pytest.mark.parametrize("field", ["alt_text"])
def test_frontmatter_rejects_empty_real_registry_alternatives(field: str) -> None:
    metadata = FIGURE_METADATA["frontmatter_model"]
    original = dict(metadata)
    try:
        metadata[field] = ""
        with pytest.raises(ValueError, match="missing frontmatter alternative"):
            validate_frontmatter(ROOT)
    finally:
        metadata.clear()
        metadata.update(original)
    assert metadata == original
    validate_frontmatter(ROOT)


def test_frontmatter_claims_match_owners_and_actual_recovery() -> None:
    from figures.frontmatter_execution import build_figure as execution
    from figures.frontmatter_robustness import READING
    from figures.frontmatter_robustness import build_figure as robustness

    figures = [robustness(), execution()]
    try:
        robust_text = " ".join(t.get_text() for t in figures[0].findobj(Text))
        execution_text = " ".join(t.get_text() for t in figures[1].findobj(Text))
        assert "Recovery only" in robust_text
        assert "No client theorem transfers to either server rule." in robust_text
        assert "No estimator B-robustness" in robust_text
        assert "Parser success ≠ execution" in execution_text
        assert "Integrity ≠ scientific validity" in execution_text
        assert "No backend certified here" in execution_text
        methods = (ROOT / "manuscript/07_methods_aggregation.md").read_text()
        assert "does not\ntransfer FedGVI's client theorem" in methods
        assert "not a reconstruction" in methods
        assert "raw effective-weight bound" in methods
        # Projection readings are an independent emitted surface. Check their
        # ownership language as well as the canonical diagram's actual artists.
        reading = " ".join(READING)
        assert "Recovery only; no transferred client theorem" in reading
        assert "guaranteed universal robustness" not in reading.lower()
    finally:
        for fig in figures:
            plt.close(fig)
    # Real asymmetric beliefs: recovery passes while a changed robustness has
    # an observable effect. A constant or empty oracle cannot pass this control.
    beliefs = np.array([[0.8, 0.2], [0.7, 0.3], [0.01, 0.99]])
    reference = log_linear_pool(beliefs)
    assert np.array_equal(robust_aggregate(beliefs, robustness=0).consensus, reference)
    assert not np.allclose(robust_aggregate(beliefs, robustness=2).consensus, reference)


def test_gnn_native_consumes_model_values_and_rejects_changed_expectation() -> None:
    from fedference.gnn.loader import load_sentinel
    from fedference.gnn.native import native_posterior, native_world
    from fedference.pomdp import build_sentinel_world

    model = load_sentinel()
    world = native_world(model)
    expected = build_sentinel_world()
    for key in ("A", "B", "C", "D"):
        assert np.array_equal(world[key], expected[key])
    prior = np.log(np.full(np.asarray(world["D"]).size, 1 / np.asarray(world["D"]).size))
    posterior = native_posterior(model, 0, prior)
    assert np.isclose(posterior.sum(), 1)
    changed = prior.copy()
    changed[0] += 1
    assert not np.allclose(native_posterior(model, 0, changed), posterior)
