"""Roadmap registry source, profile, and dataset integrity."""

from __future__ import annotations

import json

import pytest

from fedference.research_registry import (
    BNN_PROTOCOL_PROFILES,
    DATASET_SPECS,
    EXECUTION_PROFILES,
    EXPERIMENT_SPECS,
    SOURCE_REFERENCES,
    get_dataset_spec,
    get_experiment_spec,
    registry_fingerprint,
    registry_manifest,
)


def test_registry_manifest_is_complete_deterministic_and_json_safe() -> None:
    first = registry_manifest()
    second = registry_manifest()
    assert first == second
    assert json.loads(json.dumps(first)) == first
    assert registry_fingerprint() == registry_fingerprint()
    assert len(registry_fingerprint()) == 64
    assert len(DATASET_SPECS) == 3
    assert len({spec.dataset_id for spec in DATASET_SPECS}) == len(DATASET_SPECS)
    assert len({spec.experiment_id for spec in EXPERIMENT_SPECS}) == len(EXPERIMENT_SPECS)
    assert {spec.dataset_id for spec in DATASET_SPECS} == {
        "uci-wdbc",
        "uci-dry-bean",
        "uci-banknote",
    }
    assert {spec.experiment_id for spec in EXPERIMENT_SPECS} >= {
        "server-theory",
        "fedgvi-bnn",
        "external-tabular",
        "friston-protocol",
        "hybrid-tracking",
        "hierarchy-tasks",
        "multi-node-emulator",
    }
    assert all(spec.smallest_effect_of_interest for spec in EXPERIMENT_SPECS)
    assert all(spec.mcse_stopping_target for spec in EXPERIMENT_SPECS)
    assert all(spec.maximum_budget for spec in EXPERIMENT_SPECS)
    assert all(spec.comparison_family for spec in EXPERIMENT_SPECS)
    assert not any(spec.confirmatory_ready for spec in EXPERIMENT_SPECS)
    assert all(spec.schema and "split_sha256" in spec.split_policy for spec in DATASET_SPECS)
    assert all(profile in EXECUTION_PROFILES for spec in EXPERIMENT_SPECS for profile in spec.profiles)
    assert set(BNN_PROTOCOL_PROFILES) <= set(EXECUTION_PROFILES)


def test_bnn_profiles_preserve_source_scale_without_local_execution_claim() -> None:
    source = BNN_PROTOCOL_PROFILES["source_5090"]
    m4 = BNN_PROTOCOL_PROFILES["m4_confirmatory"]
    assert source["max_local_epochs"] == 2500
    assert source["posterior_predictive_samples"] == 200
    assert source["elbo_samples"] == 10
    assert source["executed_locally"] is False
    assert source["source_seed_table"] == [42, 676, 93, 215, 318, 242]
    assert source["source_run_indices"] == [1, 2, 3, 4, 5]
    assert source["seeds"] == [676, 93, 215, 318, 242]
    assert m4["seeds"] == source["seeds"]
    assert source["client_split"] == "homogeneous"
    assert source["early_stopping_patience"] == 10
    assert source["network"] == {
        "type": "fc",
        "hidden_layers": 2,
        "hidden_width": 100,
    }
    for field in (
        "client_split",
        "network",
        "fedgvi_client_divergence",
        "fedgvi_loss_parameters",
        "pvi_client_divergence",
        "pvi_loss",
        "early_stopping_patience",
    ):
        assert m4[field] == source[field]
    assert EXECUTION_PROFILES["m4_confirmatory"]["publication_evidence"] is True
    assert EXECUTION_PROFILES["source_5090"]["publication_evidence"] is False


def test_registry_lookups_fail_closed() -> None:
    assert get_experiment_spec("server-theory").runner == "heuristic-characterization"
    assert get_dataset_spec("uci-banknote").license == "CC BY 4.0"
    with pytest.raises(KeyError, match="unknown experiment_id"):
        get_experiment_spec("missing")
    with pytest.raises(KeyError, match="unknown dataset_id"):
        get_dataset_spec("missing")


def test_multi_node_emulator_is_bound_to_transport_authorities() -> None:
    source_ids = {source.source_id for source in SOURCE_REFERENCES}
    emulator_sources = set(get_experiment_spec("multi-node-emulator").source_ids)

    assert {
        "ietf-rfc8446-tls13",
        "ietf-rfc5280-pki",
        "python-ssl",
        "docker-compose-networking",
        "docker-engine-security",
    } <= emulator_sources <= source_ids
