"""Source-bound registry for the Active Fedference research programme.

Registry entries declare intended evidence; they are not results. An
``active`` or ``planned`` entry therefore never implies that its falsifier has
passed or that a manuscript claim has been earned.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .evidence import (
    DatasetSpec,
    ExperimentSpec,
    SourceReference,
    canonical_sha256,
)

SOURCE_REFERENCES: tuple[SourceReference, ...] = (
    SourceReference(
        source_id="mildner-2025-fedgvi",
        title="Federated Generalised Variational Inference",
        url="https://proceedings.mlr.press/v267/mildner25a.html",
        role="authoritative FedGVI method and experiment source",
    ),
    SourceReference(
        source_id="fedgvi-source",
        title="FedGVI public implementation",
        url="https://github.com/Terje-M/FedGVI",
        role="authoritative implementation protocol source",
        revision="5440352890037a81218285b8f4de81090861e9df",
    ),
    SourceReference(
        source_id="friston-2024-belief-sharing",
        title="Federated inference and belief sharing",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC11139662/",
        doi="10.1016/j.neubiorev.2023.105500",
        role="authoritative federated active-inference source",
    ),
    SourceReference(
        source_id="nguyen-2026-closed-form-gvi",
        title="Closed-form solutions for generalized variational inference",
        url="https://arxiv.org/abs/2606.25492",
        role="preprint design input; not proof for this repository",
    ),
    SourceReference(
        source_id="rangarajan-2026-hierarchical",
        title="Hierarchical Active Inference using Successor Representations",
        url="https://arxiv.org/abs/2604.15679",
        role="preprint task-family design input",
    ),
    SourceReference(
        source_id="ietf-rfc8446-tls13",
        title="The Transport Layer Security (TLS) Protocol Version 1.3",
        url="https://www.rfc-editor.org/rfc/rfc8446.html",
        role=(
            "authoritative confidentiality, integrity, and peer-authentication "
            "protocol source for the planned mTLS emulator"
        ),
    ),
    SourceReference(
        source_id="ietf-rfc5280-pki",
        title="Internet X.509 Public Key Infrastructure Certificate and CRL Profile",
        url="https://www.rfc-editor.org/rfc/rfc5280.html",
        role=(
            "authoritative certificate-path and revocation profile source for "
            "the planned mTLS trust boundary"
        ),
    ),
    SourceReference(
        source_id="python-ssl",
        title="Python ssl — TLS/SSL wrapper for socket objects",
        url="https://docs.python.org/3/library/ssl.html",
        role="standard-library implementation source for the planned TLS adapter",
    ),
    SourceReference(
        source_id="docker-compose-networking",
        title="Docker Compose networking",
        url="https://docs.docker.com/compose/how-tos/networking/",
        role="authoritative local-container network and service-discovery source",
    ),
    SourceReference(
        source_id="docker-engine-security",
        title="Docker Engine security",
        url="https://docs.docker.com/engine/security/",
        role="container-host trust-boundary and daemon-exposure design source",
    ),
)

DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        dataset_id="uci-wdbc",
        name="Breast Cancer Wisconsin Diagnostic",
        source_url=("https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"),
        doi="10.24432/C5DW2B",
        license="CC BY 4.0",
        archive_sha256="bc154869ef13f753f9e2b5a17e248cfe1ba4b6721db7c4da9f4880e40b05d3af",
        archive_member="wdbc.data",
        file_format="csv",
        n_rows=569,
        n_features=30,
        n_classes=2,
        has_missing_values=False,
        preprocessing=(
            "drop the non-predictive sample identifier",
            "encode diagnosis B=0 and M=1",
            "fit standardization on each training split only",
        ),
        schema=(
            "sample_id: source identifier excluded from predictors",
            "diagnosis: categorical target B or M",
            "features: 30 float64 diagnostic measurements in source order",
        ),
        split_policy=(
            "seeded 70/30 permutation; fit z-score on training rows only; each run records its split_sha256"
        ),
    ),
    DatasetSpec(
        dataset_id="uci-dry-bean",
        name="Dry Bean",
        source_url=("https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"),
        doi="10.24432/C50S4B",
        license="CC BY 4.0",
        archive_sha256="0a64eff5be87f48c3dbbfc0a12a56c5d5b5167ef8e61cd45d69b3e7c7130c06f",
        archive_member="DryBeanDataset/Dry_Bean_Dataset.arff",
        file_format="arff",
        n_rows=13611,
        n_features=16,
        n_classes=7,
        has_missing_values=False,
        preprocessing=(
            "parse the declared ARFF data section",
            "encode class labels in lexical order",
            "fit standardization on each training split only",
        ),
        schema=(
            "features: 16 float64 morphometric measurements in ARFF order",
            "Class: seven-class categorical target encoded in lexical order",
        ),
        split_policy=(
            "seeded 70/30 permutation; fit z-score on training rows only; each run records its split_sha256"
        ),
    ),
    DatasetSpec(
        dataset_id="uci-banknote",
        name="Banknote Authentication",
        source_url=("https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip"),
        doi="10.24432/C55P57",
        license="CC BY 4.0",
        archive_sha256="1e2acd9a2085fadf3d8145c12d3d22af853320d52294a6590c2eaf75fdc05227",
        archive_member="data_banknote_authentication.txt",
        file_format="csv",
        n_rows=1372,
        n_features=4,
        n_classes=2,
        has_missing_values=False,
        preprocessing=(
            "parse four numeric features and the integer class",
            "retain the declared 0/1 class encoding",
            "fit standardization on each training split only",
        ),
        schema=(
            "features: variance, skewness, curtosis, and entropy as float64",
            "class: binary integer target 0 or 1",
        ),
        split_policy=(
            "seeded 70/30 permutation; fit z-score on training rows only; each run records its split_sha256"
        ),
    ),
)

EXPERIMENT_SPECS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        experiment_id="server-theory",
        version="0.2",
        title="Scoped server-objective no-go and calibrated robustness",
        state="active",
        source_ids=(
            "mildner-2025-fedgvi",
            "friston-2024-belief-sharing",
            "nguyen-2026-closed-form-gvi",
        ),
        primary_estimand=(
            "truth value of the declared separable raw-log-pool no-go proposition"
        ),
        independent_unit=(
            "analytic interior-simplex construction; empirical grid rows are separate diagnostics"
        ),
        falsifier=(
            "an in-class objective with the stated all-interior q block, or a "
            "counterexample to the construction's stationarity contradiction"
        ),
        no_claim=(
            "the scoped no-go does not make robust_aggregate objective-backed, "
            "universally robust, or devoid of every broader objective construction"
        ),
        profiles=("smoke", "pilot", "confirmatory"),
        smallest_effect_of_interest=(
            "not applicable to the formal proposition; empirical follow-up must predeclare its own threshold"
        ),
        mcse_stopping_target=(
            "not applicable to the deterministic proof; empirical follow-up must "
            "freeze a world-level MCSE target"
        ),
        maximum_budget=(
            "deterministic witness plus a separately predeclared empirical-grid budget"
        ),
        comparison_family=(
            "primary proper-score family: naive, robust, variational, linear, and CLR median; "
            "secondary calibration and runtime contrasts corrected as one family"
        ),
        runner="heuristic-characterization",
    ),
    ExperimentSpec(
        experiment_id="robustness-calibration",
        version="0.2",
        title="Leakage-free robustness calibration pilot",
        state="active",
        source_ids=("mildner-2025-fedgvi", "friston-2024-belief-sharing"),
        primary_estimand="mean held-out log score over independent calibration worlds",
        independent_unit="calibration world; agents and states are nested",
        falsifier="calibration/evaluation overlap or a changed frozen configuration",
        no_claim=(
            "calibration separates tuning from evaluation but does not make the "
            "server heuristic objective-backed or universally robust"
        ),
        profiles=("smoke", "pilot", "confirmatory"),
        smallest_effect_of_interest="pilot freezes the proper-score effect threshold",
        mcse_stopping_target="pilot freezes the world-level MCSE target",
        maximum_budget="disjoint calibration and evaluation worlds with complete candidate scores",
        comparison_family="naive and robustness candidates scored by held-out log score",
        runner="calibration",
    ),
    ExperimentSpec(
        experiment_id="fedgvi-bnn",
        version="0.3",
        title="Protocol-faithful mean-field FedGVI BNN",
        state="active",
        source_ids=("mildner-2025-fedgvi", "fedgvi-source"),
        primary_estimand=("paired held-out log-score difference between FedGVI and matched PVI/NLL"),
        independent_unit="independently seeded end-to-end training run",
        falsifier="the locked proper-score interval includes zero or reverses",
        no_claim=(
            "portable MPS evidence is not exact source-scale CUDA replication "
            "and does not transfer a server theorem"
        ),
        profiles=("smoke", "pilot", "m4_confirmatory", "source_5090"),
        smallest_effect_of_interest=(
            "pilot must freeze a paired held-out log-score threshold for m4_confirmatory"
        ),
        mcse_stopping_target=("pilot must freeze a seed-level MCSE target before m4_confirmatory"),
        maximum_budget=(
            "M4 rounds and local epochs are frozen by pilot; source_5090 retains the "
            "source maxima declaratively and is not executed locally"
        ),
        comparison_family=(
            "FedGVI versus matched PVI/NLL is primary; dataset, accuracy, and ECE contrasts are secondary"
        ),
        runner="fedgvi-bnn",
    ),
    ExperimentSpec(
        experiment_id="external-tabular",
        version="0.3",
        title="Three-dataset UCI benchmark pack",
        state="active",
        source_ids=("mildner-2025-fedgvi",),
        primary_estimand="contamination-conditioned held-out log-score difference",
        independent_unit="licensed external dataset, with seeds nested",
        falsifier="non-reproducible source bytes or a null/reversed locked contrast",
        no_claim="three datasets do not establish universal deployment validity",
        profiles=("smoke", "pilot", "confirmatory"),
        smallest_effect_of_interest=("pilot must freeze a per-dataset held-out log-score threshold"),
        mcse_stopping_target=(
            "pilot must freeze nested-seed MCSE targets without treating seeds as datasets"
        ),
        maximum_budget=("three pinned datasets; seed count and contamination grid frozen after pilot"),
        comparison_family=(
            "per-dataset naive, robust, and variational proper-score contrasts; "
            "accuracy and ECE are secondary"
        ),
        runner="external-tabular",
    ),
    ExperimentSpec(
        experiment_id="friston-protocol",
        version="0.4",
        title="Friston Eq. 2 and Figures 5, 7, and 9 reconstruction",
        state="planned",
        source_ids=("friston-2024-belief-sharing",),
        primary_estimand="source-defined plotted quantity in native units",
        independent_unit="source-defined agent, episode, or seed",
        falsifier="a protocol negative control fails its expected direction or null",
        no_claim=(
            "paper-constrained reconstruction is not exact replication while "
            "source parameters or routines remain unresolved"
        ),
        profiles=("smoke", "pilot", "confirmatory"),
        smallest_effect_of_interest=(
            "source-defined numerical tolerance must be extracted before confirmatory execution"
        ),
        mcse_stopping_target=("source-defined stochastic unit and precision target must be extracted first"),
        maximum_budget=("bounded Python reconstruction budget, frozen after the parity audit"),
        comparison_family=("Equation 2 and Figures 5, 7, and 9 are separate source-parity targets"),
        runner="friston-protocol",
    ),
    ExperimentSpec(
        experiment_id="hybrid-tracking",
        version="0.4",
        title="Discrete-context continuous target tracking",
        state="planned",
        source_ids=("friston-2024-belief-sharing",),
        primary_estimand="held-out posterior-predictive log score",
        independent_unit="independently seeded tracking world",
        falsifier="a categorical, Gaussian, or zero-robustness recovery gate fails",
        no_claim="a minimal tracking task does not establish general continuous control",
        profiles=("smoke", "pilot", "confirmatory"),
        smallest_effect_of_interest=("pilot must freeze a posterior-predictive log-score threshold"),
        mcse_stopping_target=("pilot must freeze the seeded-world MCSE target"),
        maximum_budget=("tracking-world count and horizon are frozen after recovery-gate pilot"),
        comparison_family=(
            "naive and robust hybrid against discrete-only, continuous-only, and oracle-context controls"
        ),
        runner="hybrid-tracking",
    ),
    ExperimentSpec(
        experiment_id="hierarchy-tasks",
        version="0.4",
        title="Partially observed Four Rooms and Key-Door hierarchy study",
        state="planned",
        source_ids=(
            "friston-2024-belief-sharing",
            "rangarajan-2026-hierarchical",
        ),
        primary_estimand="episode success within a fixed horizon",
        independent_unit="task, with seeds and episodes nested",
        falsifier="learned hierarchy fails to beat matched flat and shuffled controls",
        no_claim="a task-specific gain is not a general hierarchy theorem",
        profiles=("smoke", "pilot", "confirmatory"),
        smallest_effect_of_interest=("pilot must freeze a task-level success-probability threshold"),
        mcse_stopping_target=("pilot must freeze per-task MCSE with seeds and episodes nested"),
        maximum_budget=("matched horizon and compute cap across Four Rooms and Key-Door"),
        comparison_family=(
            "flat, oracle hierarchy, learned hierarchy, shuffled hierarchy, and non-gating hierarchy"
        ),
        runner="hierarchy-tasks",
    ),
    ExperimentSpec(
        experiment_id="multi-node-emulator",
        version="1.0",
        title="Authenticated local multi-node federation emulator",
        state="planned",
        source_ids=(
            "mildner-2025-fedgvi",
            "ietf-rfc8446-tls13",
            "ietf-rfc5280-pki",
            "python-ssl",
            "docker-compose-networking",
            "docker-engine-security",
        ),
        primary_estimand="consensus identity and declared fault outcome per round",
        independent_unit="replicated local container federation round",
        falsifier="tamper, replay, timeout, or wrong-key control does not fail closed",
        no_claim="local containers are not physical multi-host deployment",
        profiles=("smoke", "confirmatory"),
        smallest_effect_of_interest=("zero consensus divergence and exact declared fault disposition"),
        mcse_stopping_target=("not statistical; deterministic faults must reproduce across declared rounds"),
        maximum_budget=("container count, round count, and fault schedule frozen in the emulator profile"),
        comparison_family=(
            "drop, duplicate, delay, replay, tamper, timeout, restart, and out-of-order controls"
        ),
    ),
)

EXECUTION_PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {
        "publication_evidence": False,
        "purpose": "bounded correctness feedback only",
        "device": "cpu",
    },
    "pilot": {
        "publication_evidence": False,
        "purpose": "lock calibration, budget, smallest effect, and MCSE target",
        "device": "cpu-or-mps",
    },
    "confirmatory": {
        "publication_evidence": True,
        "purpose": "locked source-bound evaluation",
        "device": "declared-in-receipt",
    },
    "m4_confirmatory": {
        "publication_evidence": True,
        "purpose": "protocol-faithful portable Apple M4 evaluation",
        "device": "mps-with-cpu-reference",
    },
    "source_5090": {
        "publication_evidence": False,
        "purpose": "declarative exact source-scale CUDA profile; unexecuted locally",
        "device": "cuda-rtx-5090",
    },
}

BNN_PROTOCOL_PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {
        "datasets": ["FashionMNIST"],
        "seeds": [42],
        "contamination_rates": [0.0, 0.6],
        "n_clients": 3,
        "server_rounds": 1,
        "local_epochs": 1,
        "posterior_predictive_samples": 2,
        "elbo_samples": 1,
    },
    "m4_confirmatory": {
        "datasets": ["MNIST", "FashionMNIST", "KMNIST"],
        "seeds": [676, 93, 215, 318, 242],
        "contamination_seed": 42,
        "contamination_rates": [0.0, 0.1, 0.2, 0.4, 0.6],
        "n_clients": 3,
        "client_split": "homogeneous",
        "server_rounds": 25,
        "network": {
            "type": "fc",
            "hidden_layers": 2,
            "hidden_width": 100,
        },
        "fedgvi_client_divergence": {"name": "AR", "parameter": 2.5},
        "fedgvi_loss_parameters": [1.0, 0.5],
        "pvi_client_divergence": {"name": "KLD"},
        "pvi_loss": {"name": "nll"},
        "early_stopping_patience": 10,
        "compute_budget": "locked by pilot before confirmatory execution",
        "local_epochs": "locked by pilot before confirmatory execution",
        "posterior_predictive_samples": "locked by pilot before confirmatory execution",
        "elbo_samples": "locked by pilot before confirmatory execution",
        "checkpoint_every_server_round": True,
    },
    "pilot": {
        "datasets": ["synthetic-binary"],
        "seeds": [0],
        "contamination_rates": [0.0, 0.6],
        "n_clients": 3,
        "server_rounds": 2,
        "local_epochs": 3,
        "posterior_predictive_samples": 1,
        "elbo_samples": 1,
        "executed_locally": True,
    },
    "source_5090": {
        "datasets": ["FashionMNIST"],
        "seeds": [676, 93, 215, 318, 242],
        "source_seed_table": [42, 676, 93, 215, 318, 242],
        "source_run_indices": [1, 2, 3, 4, 5],
        "contamination_seed": 42,
        "contamination_rates": [0.0, 0.1, 0.2, 0.4, 0.6],
        "n_clients": 3,
        "client_split": "homogeneous",
        "server_rounds": 25,
        "max_local_epochs": 2500,
        "early_stopping_patience": 10,
        "posterior_predictive_samples": 200,
        "elbo_samples": 10,
        "network": {
            "type": "fc",
            "hidden_layers": 2,
            "hidden_width": 100,
        },
        "fedgvi_client_divergence": {"name": "AR", "parameter": 2.5},
        "fedgvi_loss_parameters": [1.0, 0.5],
        "pvi_client_divergence": {"name": "KLD"},
        "pvi_loss": {"name": "nll"},
        "executed_locally": False,
    },
}


def _by_id(values: tuple[Any, ...], field: str) -> dict[str, Any]:
    index: dict[str, Any] = {}
    duplicates: set[str] = set()
    for value in values:
        identifier = getattr(value, field)
        if identifier in index:
            duplicates.add(identifier)
        index[identifier] = value
    if duplicates:
        raise RuntimeError(f"registry contains duplicate {field} values: {sorted(duplicates)}")
    return index


def get_experiment_spec(experiment_id: str) -> ExperimentSpec:
    """Return one declared experiment or raise a stable lookup error."""
    try:
        return _by_id(EXPERIMENT_SPECS, "experiment_id")[experiment_id]
    except KeyError as exc:
        raise KeyError(f"unknown experiment_id {experiment_id!r}") from exc


def get_dataset_spec(dataset_id: str) -> DatasetSpec:
    """Return one declared dataset or raise a stable lookup error."""
    try:
        return _by_id(DATASET_SPECS, "dataset_id")[dataset_id]
    except KeyError as exc:
        raise KeyError(f"unknown dataset_id {dataset_id!r}") from exc


def registry_manifest() -> dict[str, Any]:
    """Return the canonical machine-readable research registry."""
    source_ids = set(_by_id(SOURCE_REFERENCES, "source_id"))
    _by_id(DATASET_SPECS, "dataset_id")
    _by_id(EXPERIMENT_SPECS, "experiment_id")
    unresolved = {
        source_id
        for experiment in EXPERIMENT_SPECS
        for source_id in experiment.source_ids
        if source_id not in source_ids
    }
    if unresolved:
        raise RuntimeError(f"registry contains unresolved source ids: {sorted(unresolved)}")
    unresolved_profiles = {
        profile
        for experiment in EXPERIMENT_SPECS
        for profile in experiment.profiles
        if profile not in EXECUTION_PROFILES
    }
    unresolved_profiles.update(set(BNN_PROTOCOL_PROFILES) - set(EXECUTION_PROFILES))
    if unresolved_profiles:
        raise RuntimeError(f"registry contains unresolved execution profiles: {sorted(unresolved_profiles)}")
    manifest = {
        "schema_version": "1.0",
        "sources": [asdict(source) for source in SOURCE_REFERENCES],
        "datasets": [asdict(dataset) for dataset in DATASET_SPECS],
        "experiments": [asdict(experiment) for experiment in EXPERIMENT_SPECS],
        "execution_profiles": EXECUTION_PROFILES,
        "bnn_protocol_profiles": BNN_PROTOCOL_PROFILES,
    }
    return json.loads(json.dumps(manifest, sort_keys=True))


def registry_fingerprint() -> str:
    """Return the canonical SHA-256 of :func:`registry_manifest`."""
    return canonical_sha256(registry_manifest())


__all__ = [
    "BNN_PROTOCOL_PROFILES",
    "DATASET_SPECS",
    "EXECUTION_PROFILES",
    "EXPERIMENT_SPECS",
    "SOURCE_REFERENCES",
    "get_dataset_spec",
    "get_experiment_spec",
    "registry_fingerprint",
    "registry_manifest",
]
