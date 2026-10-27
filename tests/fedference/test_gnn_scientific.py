"""Real upstream/native tests; no mocked parser or normalized site increments."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fedference.gnn.admission import GNNAdmissionError
from fedference.gnn.scientific import (
    HYBRID_ASSET,
    SITES_ASSET,
    checkpoint,
    gaussian_condition,
    load_scientific,
    restore_checkpoint,
    run_sites,
    run_tracking,
    tracking_config,
)
from fedference.gnn.source import EXTENSION_MANIFEST, parse_extensions
from fedference.gnn.types import HybridGaussianModel, SiteFactorModel


@pytest.fixture
def upstream() -> Path:
    value = os.environ.get("GNN_ROOT")
    if value is None:
        pytest.skip("real parser gate requires explicit GNN_ROOT")
    return Path(value)


@pytest.fixture
def dependencies() -> Path:
    value = os.environ.get("GNN_PARSER_DEPENDENCIES")
    if value is None:
        pytest.skip("real normal parser import requires explicit GNN_PARSER_DEPENDENCIES")
    return Path(value)


def _asset(path: Path, source: Path, field: str, value: object) -> Path:
    prefix, payload = source.read_text().split("## FedferenceScientific\n")
    data = json.loads(payload)
    current = data
    keys = field.split(".")
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value
    path.write_text(prefix + "## FedferenceScientific\n" + json.dumps(data, indent=2) + "\n")
    return path


def test_hybrid_and_cavity_native_recovery(tmp_path: Path, upstream: Path, dependencies: Path) -> None:
    # Given actual upstream-preserved Gaussian and signed-site assets.
    hybrid = load_scientific(HYBRID_ASSET, upstream_root=upstream, dependency_root=dependencies)
    sites = load_scientific(SITES_ASSET, upstream_root=upstream, dependency_root=dependencies)
    assert isinstance(hybrid.model, HybridGaussianModel)
    assert isinstance(sites.model, SiteFactorModel)
    # When the real tracking and FedGVI native consumers execute.
    tracking = run_tracking(hybrid, seed=1943)
    run = run_sites(sites)
    posterior = gaussian_condition(hybrid.model, (3.0, 2.0), ((0.04, 0.0), (0.0, 0.01)))
    # Then every arithmetic identity has an independent numeric oracle.
    np.testing.assert_allclose(posterior.mean, (2.0, 1.0), atol=1e-14)
    np.testing.assert_allclose(posterior.covariance, ((0.02, 0.0), (0.0, 0.005)), atol=1e-14)
    np.testing.assert_array_equal(run.initial.sites[0].precision, [-0.5, 1.0])
    np.testing.assert_allclose(run.initial.posterior().mean, [0.0, 1 / 22], atol=1e-14)
    np.testing.assert_allclose(run.initial.posterior().variance, [0.5, 2 / 11], atol=1e-14)
    np.testing.assert_allclose(run.cavity.mean, [0.1, -1 / 18])
    np.testing.assert_allclose(run.cavity.variance, [0.4, 2 / 9])
    np.testing.assert_array_equal(run.updated.sites[0].precision, [1.5, -2.5])
    np.testing.assert_allclose(run.updated.posterior().mean, [0.5, 1 / 6])
    assert run.parity.exact is False
    saved = checkpoint(sites, run.updated)
    assert restore_checkpoint(sites, saved).fingerprint == run.updated.fingerprint
    print(
        json.dumps(
            {
                "tracking": tracking,
                "posterior": posterior.mean,
                "initial": run.initial.as_dict(),
                "updated": run.updated.as_dict(),
                "parity": run.parity.as_dict(),
                "checkpoint": json.loads(saved),
            }
        )
    )


def test_singular_covariance_and_site_as_pmf(tmp_path: Path, upstream: Path, dependencies: Path) -> None:
    # Given a singular covariance and an explicit PMF interpretation of a signed factor.
    singular = _asset(
        tmp_path / "singular.gnn.md", HYBRID_ASSET, "model.covariance.values", [0.04, 0.0, 0.0, 0.0]
    )
    pmf = _asset(tmp_path / "pmf.gnn.md", SITES_ASSET, "native.representation", "probability-mass")
    protected = {path: path.read_bytes() for path in (singular, pmf, HYBRID_ASSET, SITES_ASSET)}
    # When actual upstream parsing reaches project admission.
    with pytest.raises(GNNAdmissionError, match="GNN_COVARIANCE") as covariance:
        load_scientific(singular, upstream_root=upstream, dependency_root=dependencies)
    with pytest.raises(GNNAdmissionError, match="GNN_SITE_AS_PMF") as signs:
        load_scientific(pmf, upstream_root=upstream, dependency_root=dependencies)
    # Then both refuse and no input is overwritten.
    assert all(path.read_bytes() == before for path, before in protected.items())
    print(json.dumps({"singular": str(covariance.value), "site_pmf": str(signs.value)}))


def test_source_mutations_change_native_values_and_bind_checkpoint(
    tmp_path: Path, upstream: Path, dependencies: Path
) -> None:
    # Given explicit changes to the Gaussian mean and signed precision.
    changed_h = _asset(tmp_path / "target.gnn.md", HYBRID_ASSET, "model.mean.values", [-2.0, 0.0])
    changed_s = _asset(
        tmp_path / "site.gnn.md", SITES_ASSET, "model.precision_increment.values", [-1.0, 0.0, 0.0, 1.0]
    )
    original_h = load_scientific(HYBRID_ASSET, upstream_root=upstream, dependency_root=dependencies)
    original_s = load_scientific(SITES_ASSET, upstream_root=upstream, dependency_root=dependencies)
    # When the modified source is parsed and executed.
    h = load_scientific(changed_h, upstream_root=upstream, dependency_root=dependencies)
    s = load_scientific(changed_s, upstream_root=upstream, dependency_root=dependencies)
    changed_tracking = run_tracking(h, seed=1943)
    changed_sites = run_sites(s)
    # Then actual trajectories and known Gaussian arithmetic change.
    assert tracking_config(h).target_position == -2
    assert changed_tracking["result"] != run_tracking(original_h, seed=1943)["result"]
    np.testing.assert_allclose(changed_sites.initial.posterior().variance, [2 / 3, 2 / 11])
    saved = checkpoint(original_s, run_sites(original_s).updated)
    with pytest.raises(GNNAdmissionError, match="GNN_CHECKPOINT_BINDING"):
        restore_checkpoint(s, saved)
    print(
        json.dumps(
            {
                "tracking": changed_tracking,
                "site": changed_sites.initial.as_dict(),
                "original_digest": original_s.digest,
                "changed_digest": s.digest,
            }
        )
    )


def test_full_covariance_recovery_and_tracking_refusal(
    tmp_path: Path, upstream: Path, dependencies: Path
) -> None:
    # Given a real GNN correlated Gaussian with a known equal-noise update.
    means = _asset(tmp_path / "means.gnn.md", HYBRID_ASSET, "model.mean.values", [2.0, 4.0])
    path = _asset(tmp_path / "correlated.gnn.md", means, "model.covariance.values", [2.0, 1.0, 1.0, 2.0])
    original = load_scientific(path, upstream_root=upstream, dependency_root=dependencies)
    assert isinstance(original.model, HybridGaussianModel)
    model = original.model
    # When full Gaussian conditioning consumes all covariance entries.
    posterior = gaussian_condition(model, (4.0, 2.0), ((2.0, 1.0), (1.0, 2.0)))
    # Then the known matrix answer holds, while narrow tracking refuses.
    np.testing.assert_allclose(posterior.mean, [3.0, 3.0], atol=1e-14)
    np.testing.assert_allclose(posterior.covariance, [[1.0, 0.5], [0.5, 1.0]], atol=1e-14)
    with pytest.raises(GNNAdmissionError, match="GNN_CAPABILITY"):
        run_tracking(replace(original, model=model), seed=1943)


@pytest.mark.parametrize(
    "field,value,operation,error",
    [
        ("model.categorical.D.values", [0.6, 0.4], "tracking", "GNN_CAPABILITY"),
        ("model.categorical.C.values", [1.0, 0.0], "tracking", "GNN_CAPABILITY"),
        ("model.mean.values", [1.0, 0.2], "tracking", "GNN_CAPABILITY"),
        ("native.tracking.horizon", True, "tracking", "GNN_NUMBER"),
        ("native.prior_covariance", [[4.0, 0.0], [0.0, 0.25]], "sites", "non-normalizable cavity"),
        ("native.units", ["m", "1"], "sites", "GNN_UNIT"),
        ("native.schedule", "sequential", "sites", "GNN_SCHEDULE"),
        ("model.precision_increment.values", [-0.5, 0.1, 0.1, 1.0], "sites", "GNN_CAPABILITY"),
    ],
)
def test_unsupported_or_improper_models_refuse(
    field: str, value: object, operation: str, error: str, tmp_path: Path, upstream: Path, dependencies: Path
) -> None:
    # Given an input outside the declared native capability.
    source = HYBRID_ASSET if operation == "tracking" else SITES_ASSET
    path = _asset(tmp_path / "bad.gnn.md", source, field, value)
    # When the real source/native boundary is used.
    with pytest.raises((GNNAdmissionError, ValueError), match=error) as refusal:
        program = load_scientific(path, upstream_root=upstream, dependency_root=dependencies)
        if operation == "tracking":
            run_tracking(program, seed=1943)
        else:
            run_sites(program)
    # Then the rejection is visible and there is no accepted result.
    print(field, str(refusal.value))


def test_real_parser_preservation_and_closed_execution(
    tmp_path: Path, upstream: Path, dependencies: Path
) -> None:
    # Given only the manifested sources plus hostile unmanifested shadow modules.
    manifest = json.loads(EXTENSION_MANIFEST.read_text())
    source_root, dependency_root = tmp_path / "source", tmp_path / "dependencies"
    for root, destination, table in (
        (upstream, source_root, manifest["files"]),
        (dependencies, dependency_root, manifest["dependencies"]),
    ):
        for relative in table:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, target)
    marker = tmp_path / "UNVERIFIED_EXECUTED"
    for path in (source_root / "typing.py", source_root / "src/typing.py", dependency_root / "json.py"):
        path.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n")
    # When the real normal-import parser executes from its closed snapshot.
    parsed = parse_extensions(
        HYBRID_ASSET,
        upstream_root=source_root,
        dependency_root=dependency_root,
        allowed_extensions=("FedferenceScientific",),
    )
    # Then extensions survive verbatim and hostile files never execute.
    expected = HYBRID_ASSET.read_text().split("## FedferenceScientific\n")[1].strip()
    assert parsed.extension("FedferenceScientific") == expected
    assert "gnn/parsers/markdown_parser.py" in parsed.modules
    assert not marker.exists()
    print(
        json.dumps(
            {
                "modules": parsed.modules,
                "source": parsed.source_digest,
                "parser": parsed.parser_digest,
                "unverified_executed": marker.exists(),
            }
        )
    )


@pytest.mark.parametrize("dependency", [False, True])
def test_closure_tamper_rejected(
    dependency: bool, tmp_path: Path, upstream: Path, dependencies: Path
) -> None:
    # Given a source or dependency closure containing an altered file.
    manifest = json.loads(EXTENSION_MANIFEST.read_text())
    table = manifest["dependencies" if dependency else "files"]
    root = dependencies if dependency else upstream
    for relative in table:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    altered = tmp_path / next(iter(table))
    altered.write_bytes(altered.read_bytes() + b"\n# changed\n")
    # When the caller selects that altered closure.
    with pytest.raises(GNNAdmissionError, match="GNN_PARSER_DRIFT") as refusal:
        parse_extensions(
            HYBRID_ASSET,
            upstream_root=upstream if dependency else tmp_path,
            dependency_root=tmp_path if dependency else dependencies,
            allowed_extensions=("FedferenceScientific",),
        )
    # Then the real executable boundary refuses before parsing.
    print(str(refusal.value))


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("duplicate-section", "GNN_SECTIONS"),
        ("unknown-section", "GNN_SECTIONS"),
        ("duplicate-json", "GNN_DUPLICATE"),
        ("missing-extension", "GNN_EXTENSION"),
        ("wrong-version", "GNN_SECTIONS"),
        ("wrong-name", "GNN_EXTRACTION_MISMATCH"),
        ("extra-native", "GNN_CAPABILITY"),
    ],
)
def test_raw_source_refusals(
    mutation: str, error: str, tmp_path: Path, upstream: Path, dependencies: Path
) -> None:
    # Given malformed or conflicting source declarations.
    document = HYBRID_ASSET.read_text()
    if mutation == "duplicate-section":
        document += "\n## ModelName\nconflict\n"
    elif mutation == "unknown-section":
        document += "\n## UnadmittedExtension\nx=1\n"
    elif mutation == "duplicate-json":
        document = document.replace(
            '"schema": "fedference.scientific/1",',
            '"schema": "fedference.scientific/1", "schema": "duplicate",',
            1,
        )
    elif mutation == "missing-extension":
        document = document.split("## FedferenceScientific")[0]
    elif mutation == "wrong-version":
        document = document.replace("GNN v1.0", "GNN v2.0")
    elif mutation == "wrong-name":
        document = document.replace("## ModelName\nhybrid-tracking-observation-model", "## ModelName\nother")
    else:
        document = document.replace(
            '"capability": "native-hybrid-tracking/1",',
            '"capability": "native-hybrid-tracking/1", "extra": true,',
        )
    path = tmp_path / "invalid.gnn.md"
    path.write_text(document)
    # When the real source boundary is called.
    with pytest.raises(GNNAdmissionError, match=error):
        load_scientific(path, upstream_root=upstream, dependency_root=dependencies)
    # Then the malformed source remains untouched.
    assert path.read_text() == document


@pytest.mark.parametrize("executable,error", [("absent", "GNN_PROCESS"), ("false", "GNN_EXTRACTION")])
def test_parser_process_refusals(
    executable: str, error: str, tmp_path: Path, upstream: Path, dependencies: Path
) -> None:
    # Given an explicitly selected missing or failing process.
    failure_binary = shutil.which("false")
    assert failure_binary is not None
    python = tmp_path / "absent" if executable == "absent" else Path(failure_binary)
    # When no real parser result can be obtained.
    with pytest.raises(GNNAdmissionError, match=error):
        parse_extensions(
            HYBRID_ASSET,
            upstream_root=upstream,
            dependency_root=dependencies,
            allowed_extensions=("FedferenceScientific",),
            python=python,
        )
    # Then there is no fallback result.


def test_explicit_dependencies_required(upstream: Path) -> None:
    # Given no explicitly selected dependency closure.
    # When parsing is requested.
    with pytest.raises(GNNAdmissionError, match="GNN_PARSER_DEPENDENCY"):
        parse_extensions(HYBRID_ASSET, upstream_root=upstream, allowed_extensions=("FedferenceScientific",))
    # Then ambient packages are never searched.


def test_checkpoint_native_settings_binding(upstream: Path, dependencies: Path) -> None:
    # Given the same admitted site with changed native client data.
    program = load_scientific(SITES_ASSET, upstream_root=upstream, dependency_root=dependencies)
    settings = json.loads(program.native_json)
    settings["client_means"] = [[0.75, -0.25], [0.0, 0.2]]
    changed = replace(program, native_json=json.dumps(settings, sort_keys=True))
    saved = checkpoint(program, run_sites(program).updated)
    # When a checkpoint from the old native configuration is restored.
    with pytest.raises(GNNAdmissionError, match="GNN_CHECKPOINT_BINDING"):
        restore_checkpoint(changed, saved)
    # Then the complete native settings participate in the digest.
    assert changed.digest != program.digest


def test_normalizing_signed_site_changes_model_and_invalidates_checkpoint(
    tmp_path: Path, upstream: Path, dependencies: Path
) -> None:
    # Given a deliberately destructive PMF-style absolute-value normalization.
    original = load_scientific(SITES_ASSET, upstream_root=upstream, dependency_root=dependencies)
    path = _asset(
        tmp_path / "normalized.gnn.md",
        SITES_ASSET,
        "model.precision_increment.values",
        [1 / 3, 0.0, 0.0, 2 / 3],
    )
    changed = load_scientific(path, upstream_root=upstream, dependency_root=dependencies)
    # When that different source attempts to reuse the original checkpoint.
    with pytest.raises(GNNAdmissionError, match="GNN_CHECKPOINT_BINDING"):
        restore_checkpoint(changed, checkpoint(original, run_sites(original).updated))
    # Then normalization cannot masquerade as the original signed model.
    np.testing.assert_array_equal(run_sites(original).initial.sites[0].precision, [-0.5, 1.0])
    assert run_sites(changed).initial.fingerprint != run_sites(original).initial.fingerprint


@pytest.mark.parametrize(
    "field,value,operation,error",
    [
        ("schema", "unknown", "tracking", "GNN_SCHEMA"),
        ("model.metadata.provenance", {}, "tracking", "GNN_FIELDS"),
        ("native.field_roles.mean", "initial-state-prior", "tracking", "GNN_UNIT"),
        ("native.capability", "unknown", "sites", "GNN_CAPABILITY"),
        ("native.prior_mean", ["0", 0], "sites", "GNN_NUMBER"),
        ("native.prior_mean", [0], "sites", "GNN_SHAPE"),
        ("native.prior_covariance", [[0.5], [0, 0.25]], "sites", "GNN_SHAPE"),
        ("native.prior_covariance", [[0.5, 0.1], [0.1, 0.25]], "sites", "GNN_CAPABILITY"),
        ("model.units", ["m", "1"], "tracking", "GNN_UNIT"),
        ("native.tracking", {}, "tracking", "GNN_FIELDS"),
        ("native.tracking.horizon", 4.0, "tracking", "GNN_NUMBER"),
        ("native.tracking.robustness", 1.0, "tracking", "GNN_CAPABILITY"),
    ],
)
def test_native_schema_and_units_refuse(
    field: str,
    value: object,
    operation: str,
    error: str,
    tmp_path: Path,
    upstream: Path,
    dependencies: Path,
) -> None:
    # Given a shape, unit, provenance, or capability violation.
    source = HYBRID_ASSET if operation == "tracking" else SITES_ASSET
    path = _asset(tmp_path / "bad.gnn.md", source, field, value)
    # When source admission and the selected native operation are requested.
    with pytest.raises(GNNAdmissionError, match=error):
        program = load_scientific(path, upstream_root=upstream, dependency_root=dependencies)
        if operation == "tracking":
            run_tracking(program, seed=1943)
        else:
            run_sites(program)
    # Then no unsupported parameter interpretation executes.


@pytest.mark.parametrize(
    "covariance,error",
    [
        (((1.0, 0.5), (0.0, 1.0)), "GNN_SYMMETRY"),
        (((1.0, 0.0), (0.0, 0.0)), "GNN_COVARIANCE"),
        (((float("nan"), 0.0), (0.0, 1.0)), "GNN_SHAPE"),
    ],
)
def test_gaussian_observation_covariance_refuses(
    covariance: tuple[tuple[float, ...], ...],
    error: str,
    upstream: Path,
    dependencies: Path,
) -> None:
    # Given an invalid observation covariance, separate from the admitted prior.
    program = load_scientific(HYBRID_ASSET, upstream_root=upstream, dependency_root=dependencies)
    assert isinstance(program.model, HybridGaussianModel)
    # When full Gaussian conditioning is requested.
    with pytest.raises(GNNAdmissionError, match=error):
        gaussian_condition(program.model, (0.0, 0.0), covariance)
    # Then the malformed likelihood cannot produce a posterior.


@pytest.mark.parametrize("operation", ["tracking", "sites"])
def test_wrong_native_family_refuses(operation: str, upstream: Path, dependencies: Path) -> None:
    # Given a valid model of the other native family.
    source = SITES_ASSET if operation == "tracking" else HYBRID_ASSET
    program = load_scientific(source, upstream_root=upstream, dependency_root=dependencies)
    # When the incompatible native adapter is called.
    with pytest.raises(GNNAdmissionError, match="GNN_KIND"):
        if operation == "tracking":
            run_tracking(program, seed=1943)
        else:
            run_sites(program)
    # Then valid representation is not mistaken for execution capability.


@pytest.mark.parametrize("mutation", ["schema", "prior", "sites"])
def test_checkpoint_content_refusals(mutation: str, upstream: Path, dependencies: Path) -> None:
    # Given a source-bound checkpoint whose schema or native prior/site table is altered.
    program = load_scientific(SITES_ASSET, upstream_root=upstream, dependency_root=dependencies)
    raw = json.loads(checkpoint(program, run_sites(program).updated))
    if mutation == "schema":
        raw["schema"] = "unknown"
    elif mutation == "prior":
        raw["state"]["prior"]["mean"] = [1.0, 0.0]
    else:
        raw["state"]["sites"].append({"precision_mean": [0.0, 0.0], "precision": [0.0, 0.0]})
    # When restoration checks the declared model contract.
    with pytest.raises(GNNAdmissionError, match="GNN_CHECKPOINT"):
        restore_checkpoint(program, json.dumps(raw))
    # Then the altered checkpoint is refused even with its original model digest.


def test_checkpoint_writer_rejects_other_prior(upstream: Path, dependencies: Path) -> None:
    from fedference.bnn_fedgvi import DiagonalGaussian, FedGVIServerState

    # Given a real native server state initialized from another prior.
    program = load_scientific(SITES_ASSET, upstream_root=upstream, dependency_root=dependencies)
    state = FedGVIServerState.initialize(DiagonalGaussian(np.ones(2), np.ones(2)), n_clients=1)
    # When serialization claims the original GNN model.
    with pytest.raises(GNNAdmissionError, match="GNN_CHECKPOINT_BINDING"):
        checkpoint(program, state)
    # Then the writer refuses that false model association.


def test_other_admitted_family_is_not_scientific_program(
    tmp_path: Path, upstream: Path, dependencies: Path
) -> None:
    # Given a valid categorical payload in the scientific namespace.
    raw = json.loads(HYBRID_ASSET.read_text().split("## FedferenceScientific\n")[1])["model"]
    for name in ("coordinates", "units", "mean", "covariance"):
        raw.pop(name)
    raw["kind"] = "categorical"
    raw["metadata"]["capabilities"] = ["fedference.gnn/categorical/1"]
    path = _asset(tmp_path / "categorical.gnn.md", HYBRID_ASSET, "model", raw)
    # When the actual parser preserves that payload for scientific admission.
    with pytest.raises(GNNAdmissionError, match="GNN_KIND"):
        load_scientific(path, upstream_root=upstream, dependency_root=dependencies)
    # Then the scientific adapter refuses a valid but different family.


def test_missing_parser_source_refuses(tmp_path: Path, dependencies: Path) -> None:
    # Given an empty caller-selected parser source root.
    # When the manifested executable closure is copied.
    with pytest.raises(GNNAdmissionError, match="GNN_PARSER_DRIFT"):
        parse_extensions(
            HYBRID_ASSET,
            upstream_root=tmp_path,
            dependency_root=dependencies,
            allowed_extensions=("FedferenceScientific",),
        )
    # Then missing files are an explicit refusal.


def test_standard_section_cannot_masquerade_as_extension(upstream: Path, dependencies: Path) -> None:
    # Given a standard ModelName section requested as a purported extension.
    # When the real parser reports the true extension table.
    with pytest.raises(GNNAdmissionError, match="GNN_EXTRACTION_MISMATCH"):
        parse_extensions(
            HYBRID_ASSET,
            upstream_root=upstream,
            dependency_root=dependencies,
            allowed_extensions=("FedferenceScientific", "ModelName"),
        )
    # Then raw section presence does not substitute for extension preservation.


def test_raw_section_disagreement_refuses(tmp_path: Path, upstream: Path, dependencies: Path) -> None:
    # Given an indented header outside this boundary's canonical syntax.
    path = tmp_path / "indented.gnn.md"
    path.write_text(HYBRID_ASSET.read_text().replace("## ModelName", "  ## ModelName"))
    # When upstream accepts the looser syntax but reports a different section map.
    with pytest.raises(GNNAdmissionError, match="GNN_EXTRACTION_MISMATCH"):
        parse_extensions(
            path,
            upstream_root=upstream,
            dependency_root=dependencies,
            allowed_extensions=("FedferenceScientific",),
        )
    # Then source preservation disagreement fails closed.


def test_missing_preserved_extension_lookup(upstream: Path, dependencies: Path) -> None:
    # Given a real parser result with one admitted namespace.
    parsed = parse_extensions(
        HYBRID_ASSET,
        upstream_root=upstream,
        dependency_root=dependencies,
        allowed_extensions=("FedferenceScientific",),
    )
    # When a caller requests a namespace that was not preserved.
    with pytest.raises(GNNAdmissionError, match="GNN_EXTENSION"):
        parsed.extension("FedferenceAbsent")
    # Then the accessor cannot manufacture extension content.


def test_original_source_bytes_bind_crlf_and_unicode(
    tmp_path: Path, upstream: Path, dependencies: Path
) -> None:
    import hashlib

    # Given semantically equivalent LF/CRLF sources and a UTF-8 annotation change.
    original = HYBRID_ASSET.read_bytes()
    variants = (
        original,
        original.replace(b"\n", b"\r\n"),
        original.replace(b"Friston et al.", "Friston et al. — café".encode()),
    )
    digests = []
    for index, payload in enumerate(variants):
        path = tmp_path / f"bytes-{index}.gnn.md"
        path.write_bytes(payload)
        # When the real parser consumes an exact byte snapshot.
        parsed = parse_extensions(
            path,
            upstream_root=upstream,
            dependency_root=dependencies,
            allowed_extensions=("FedferenceScientific",),
        )
        program = load_scientific(path, upstream_root=upstream, dependency_root=dependencies)
        # Then provenance binds original bytes, including newlines and UTF-8.
        expected = hashlib.sha256(payload).hexdigest()
        assert parsed.source_digest == expected
        assert parsed.document.encode("utf-8") == payload
        assert program.model.metadata.provenance.asset == expected
        digests.append(expected)
    assert len(set(digests)) == 3
    print(json.dumps({"original_byte_sha256": digests, "variants": ["LF", "CRLF", "Unicode"]}))


def test_two_client_cavity_has_nontrivial_peer(upstream: Path, dependencies: Path) -> None:
    # Given the real GNN-bound signed primary factor and nonzero peer.
    program = load_scientific(SITES_ASSET, upstream_root=upstream, dependency_root=dependencies)
    # When the native server constructs the complete product and client cavity.
    result = run_sites(program)
    # Then this is a nontrivial distributed cavity, not the prior-only limit.
    assert len(result.initial.sites) == 2
    assert not np.array_equal(result.cavity.mean, result.initial.prior.mean)
    np.testing.assert_allclose(result.initial.posterior().precision, [2.0, 5.5])
    np.testing.assert_allclose(result.cavity.precision, [2.5, 4.5])
    print(
        json.dumps(
            {
                "site_count": len(result.initial.sites),
                "prior": result.initial.prior.mean.tolist(),
                "cavity": result.cavity.mean.tolist(),
                "peer": result.initial.sites[1].precision.tolist(),
            }
        )
    )


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("empty", "GNN_CAPABILITY"),
        ("not-list", "GNN_FIELDS"),
        ("units", "GNN_UNIT"),
        ("protocol", "GNN_PROTOCOL"),
        ("kind", "GNN_KIND"),
        ("correlation", "GNN_CAPABILITY"),
        ("client-count", "GNN_SHAPE"),
        ("client-correlation", "GNN_CAPABILITY"),
    ],
)
def test_peer_contracts_refuse(
    mutation: str, error: str, tmp_path: Path, upstream: Path, dependencies: Path
) -> None:
    # Given canonical peer declarations with one invalid relationship or native update.
    peers = json.loads(SITES_ASSET.read_text().split("## FedferenceScientific\n")[1])["peers"]
    field = "peers"
    value = peers
    if mutation == "empty":
        value = []
    elif mutation == "not-list":
        value = {}
    elif mutation == "units":
        peers[0]["units"] = ["m", "1"]
    elif mutation == "protocol":
        peers[0]["protocol"] = "probability-mass/1"
    elif mutation == "kind":
        value = [json.loads(HYBRID_ASSET.read_text().split("## FedferenceScientific\n")[1])["model"]]
    elif mutation == "correlation":
        peers[0]["precision_increment"]["values"] = [0.5, 0.1, 0.1, 0.5]
    elif mutation == "client-count":
        field, value = "native.client_means", [[0.5, -0.25]]
    else:
        field, value = "native.client_covariances", [[[0.25, 0.1], [0.1, 0.5]], [[0.5, 0.0], [0.0, 0.2]]]
    path = _asset(tmp_path / "invalid-peer.gnn.md", SITES_ASSET, field, value)
    # When real source admission and native execution are attempted.
    with pytest.raises(GNNAdmissionError, match=error):
        run_sites(load_scientific(path, upstream_root=upstream, dependency_root=dependencies))
    # Then no peer coordinate, protocol, correlation, or client update is silently discarded.


def test_hybrid_rejects_site_peers(tmp_path: Path, upstream: Path, dependencies: Path) -> None:
    # Given a valid site peer attached to the wrong native family.
    peer = json.loads(SITES_ASSET.read_text().split("## FedferenceScientific\n")[1])["model"]
    path = _asset(tmp_path / "hybrid-peer.gnn.md", HYBRID_ASSET, "peers", [peer])
    # When the hybrid namespace is admitted.
    with pytest.raises(GNNAdmissionError, match="GNN_CAPABILITY"):
        load_scientific(path, upstream_root=upstream, dependency_root=dependencies)
    # Then tracking cannot accept unused peer declarations.


def test_proper_product_with_improper_peer_cavity_refuses(
    tmp_path: Path, upstream: Path, dependencies: Path
) -> None:
    # Given a proper global product whose second client's cavity has negative precision.
    first = _asset(
        tmp_path / "negative.gnn.md", SITES_ASSET, "model.precision_increment.values", [-3.0, 0.0, 0.0, 1.0]
    )
    peers = json.loads(SITES_ASSET.read_text().split("## FedferenceScientific\n")[1])["peers"]
    peers[0]["precision_increment"]["values"] = [2.0, 0.0, 0.0, 0.5]
    path = _asset(tmp_path / "improper-cavity.gnn.md", first, "peers", peers)
    # When the real FedGVI server validates every cavity.
    with pytest.raises(ValueError, match="non-normalizable cavity") as refusal:
        run_sites(load_scientific(path, upstream_root=upstream, dependency_root=dependencies))
    # Then global properness alone cannot certify the distributed update.
    print(str(refusal.value))


@pytest.mark.parametrize("noise", [1e-20, 1e-300])
def test_gaussian_condition_retains_small_positive_covariance(
    noise: float, upstream: Path, dependencies: Path
) -> None:
    model = load_scientific(HYBRID_ASSET, upstream_root=upstream, dependency_root=dependencies).model
    assert isinstance(model, HybridGaussianModel)
    actual = gaussian_condition(model, (0.0, 0.0), ((noise, 0.0), (0.0, noise)))
    expected = np.diag([0.04 * noise / (0.04 + noise), 0.01 * noise / (0.01 + noise)])
    np.testing.assert_allclose(actual.covariance, expected, rtol=1e-14, atol=0)
    np.testing.assert_allclose(actual.mean, [noise / (0.04 + noise), 0], rtol=1e-14, atol=0)
    assert np.linalg.eigvalsh(actual.covariance).min() > 0


@pytest.mark.parametrize("variance", [1.0, 1e308])
def test_gaussian_condition_avoids_representable_mean_and_sum_overflow(
    variance: float, upstream: Path, dependencies: Path
) -> None:
    model = load_scientific(HYBRID_ASSET, upstream_root=upstream, dependency_root=dependencies).model
    assert isinstance(model, HybridGaussianModel)
    model = replace(model, mean=replace(model.mean, values=(1e308, -1e308)),
                    covariance=replace(model.covariance, values=(variance, 0, 0, variance)))
    actual = gaussian_condition(model, (-1e308, 1e308), ((variance, 0), (0, variance)))
    np.testing.assert_array_equal(actual.mean, [0, 0])
    np.testing.assert_allclose(actual.covariance, np.eye(2) * variance / 2, rtol=1e-14, atol=0)


def test_gaussian_observation_overflow_is_typed(upstream: Path, dependencies: Path) -> None:
    model = load_scientific(HYBRID_ASSET, upstream_root=upstream, dependency_root=dependencies).model
    assert isinstance(model, HybridGaussianModel)
    with pytest.raises(GNNAdmissionError, match="GNN_SHAPE"):
        gaussian_condition(model, (10**400, 0), ((1.0, 0), (0, 1.0)))


def test_gaussian_noncommuting_covariances_known_answer(
    tmp_path: Path, upstream: Path, dependencies: Path
) -> None:
    means = _asset(tmp_path / "means.gnn.md", HYBRID_ASSET, "model.mean.values", [2.0, 4.0])
    path = _asset(tmp_path / "correlated.gnn.md", means, "model.covariance.values", [2.0, 1.0, 1.0, 2.0])
    model = load_scientific(path, upstream_root=upstream, dependency_root=dependencies).model
    assert isinstance(model, HybridGaussianModel)
    actual = gaussian_condition(model, (4.0, 2.0), ((1.0, 0.0), (0.0, 4.0)))
    np.testing.assert_allclose(actual.mean, [54 / 17, 66 / 17], rtol=1e-14, atol=0)
    np.testing.assert_allclose(actual.covariance, np.array([[11, 4], [4, 20]]) / 17, rtol=1e-14, atol=0)
