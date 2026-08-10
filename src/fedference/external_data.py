"""Pinned UCI dataset acquisition and deterministic preprocessing (MAJ-6).

Archives are downloaded only into a caller-provided cache, verified against the
registry SHA-256, and parsed in memory from one declared member. Raw external
data are never silently added to the committed reviewer snapshot.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .evidence import DatasetSpec, sha256_file
from .research_registry import get_dataset_spec

ArrayF = np.ndarray
_MAX_DOWNLOAD_BYTES = 128 * 1024 * 1024
_USER_AGENT = "Active-Fedference/0.1 dataset-reproducibility-client"


@dataclass(frozen=True)
class ExternalDataset:
    """Parsed external dataset plus byte-level provenance."""

    spec: DatasetSpec
    features: ArrayF
    labels: ArrayF
    label_mapping: tuple[tuple[str, int], ...]
    archive_sha256: str
    member_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.spec, DatasetSpec):
            raise ValueError("spec must be a DatasetSpec")
        if not isinstance(self.label_mapping, (tuple, list)):
            raise ValueError("label_mapping must be a sequence")
        mapping = tuple(tuple(item) for item in self.label_mapping)
        if (
            len(mapping) != self.spec.n_classes
            or any(
                len(item) != 2
                or not isinstance(item[0], str)
                or not item[0]
                or isinstance(item[1], bool)
                or not isinstance(item[1], (int, np.integer))
                for item in mapping
            )
            or len({item[0] for item in mapping}) != len(mapping)
            or {item[1] for item in mapping} != set(range(self.spec.n_classes))
        ):
            raise ValueError("label_mapping must uniquely cover every declared class")
        raw_features = np.asarray(self.features)
        raw_labels = np.asarray(self.labels)
        if (
            np.issubdtype(raw_features.dtype, np.bool_)
            or not np.issubdtype(raw_features.dtype, np.number)
            or np.issubdtype(raw_features.dtype, np.complexfloating)
        ):
            raise ValueError("features must be real numeric values")
        if (
            np.issubdtype(raw_labels.dtype, np.bool_)
            or not np.issubdtype(raw_labels.dtype, np.number)
            or np.issubdtype(raw_labels.dtype, np.complexfloating)
        ):
            raise ValueError("labels must be real integer-valued class identifiers")
        features = np.array(raw_features, dtype=np.float64, copy=True)
        label_values = np.array(raw_labels, dtype=np.float64, copy=True)
        if features.shape != (self.spec.n_rows, self.spec.n_features):
            raise ValueError("features must match the declared dataset shape")
        if label_values.shape != (self.spec.n_rows,):
            raise ValueError("labels must match the declared dataset row count")
        if not np.all(np.isfinite(features)):
            raise ValueError("features must be finite")
        if not np.all(np.isfinite(label_values)) or not np.all(label_values == np.floor(label_values)):
            raise ValueError("labels must be finite integer-valued class identifiers")
        labels = label_values.astype(np.int64)
        if not np.array_equal(np.unique(labels), np.arange(self.spec.n_classes)):
            raise ValueError("labels must be a contiguous declared class encoding")
        for name in ("archive_sha256", "member_sha256"):
            digest = getattr(self, name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdefABCDEF" for character in digest)
            ):
                raise ValueError(f"{name} must be a SHA-256 digest")
        features.setflags(write=False)
        labels.setflags(write=False)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "label_mapping", mapping)


def _download_archive(spec: DatasetSpec, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{spec.dataset_id}-{spec.archive_sha256[:12]}.zip"
    if target.exists():
        if sha256_file(target) != spec.archive_sha256:
            raise ValueError(f"cached archive digest mismatch: {target}")
        return target
    descriptor, temporary_name = tempfile.mkstemp(
        dir=cache_dir,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    request = urllib.request.Request(spec.source_url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            total = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_DOWNLOAD_BYTES:
                    raise ValueError(f"dataset archive exceeds {_MAX_DOWNLOAD_BYTES} bytes")
                out.write(chunk)
    except (OSError, ValueError):
        temporary.unlink(missing_ok=True)
        raise
    if sha256_file(temporary) != spec.archive_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"downloaded archive digest mismatch for {spec.dataset_id}")
    temporary.replace(target)
    return target


def _read_member(archive_path: Path, spec: DatasetSpec) -> bytes:
    if sha256_file(archive_path) != spec.archive_sha256:
        raise ValueError(f"archive digest mismatch for {spec.dataset_id}")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            if spec.archive_member not in names:
                raise ValueError(f"archive member {spec.archive_member!r} is missing")
            info = archive.getinfo(spec.archive_member)
            if info.file_size > _MAX_DOWNLOAD_BYTES:
                raise ValueError("dataset member exceeds the size guard")
            return archive.read(info)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"invalid ZIP archive: {archive_path}") from exc


def _parse_wdbc(data: bytes) -> tuple[ArrayF, ArrayF, tuple[tuple[str, int], ...]]:
    rows = [row for row in csv.reader(io.StringIO(data.decode("utf-8"))) if row]
    if not rows or any(len(row) != 32 for row in rows):
        raise ValueError("WDBC member must contain 32 fields per non-empty row")
    features = np.asarray([[float(value) for value in row[2:]] for row in rows])
    mapping = (("B", 0), ("M", 1))
    label_ids = dict(mapping)
    try:
        labels = np.asarray([label_ids[row[1]] for row in rows], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"WDBC member contains unknown diagnosis {exc.args[0]!r}") from exc
    return features, labels, mapping


def _parse_banknote(data: bytes) -> tuple[ArrayF, ArrayF, tuple[tuple[str, int], ...]]:
    rows = [row for row in csv.reader(io.StringIO(data.decode("utf-8"))) if row]
    if not rows or any(len(row) != 5 for row in rows):
        raise ValueError("Banknote member must contain five fields per non-empty row")
    features = np.asarray([[float(value) for value in row[:4]] for row in rows])
    raw_labels = np.asarray([float(row[4]) for row in rows], dtype=np.float64)
    if (
        not np.all(np.isfinite(raw_labels))
        or not np.all(raw_labels == np.floor(raw_labels))
        or not set(raw_labels.astype(int).tolist()).issubset({0, 1})
    ):
        raise ValueError("Banknote labels must be integer values in {0, 1}")
    labels = raw_labels.astype(np.int64)
    return features, labels, (("0", 0), ("1", 1))


def _parse_arff(data: bytes) -> tuple[ArrayF, ArrayF, tuple[tuple[str, int], ...]]:
    data_rows: list[list[str]] = []
    in_data = False
    for raw_line in data.decode("utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue
        if line.lower() == "@data":
            in_data = True
            continue
        if in_data:
            data_rows.extend(csv.reader([line]))
    if not data_rows:
        raise ValueError("ARFF archive member contains no data rows")
    width = len(data_rows[0])
    if width < 2 or any(len(row) != width for row in data_rows):
        raise ValueError("ARFF data rows must have a consistent feature/class width")
    labels_text = [row[-1].strip() for row in data_rows]
    classes = sorted(set(labels_text))
    mapping = tuple((label, index) for index, label in enumerate(classes))
    label_ids = dict(mapping)
    features = np.asarray(
        [[float(value) for value in row[:-1]] for row in data_rows],
        dtype=np.float64,
    )
    labels = np.asarray([label_ids[label] for label in labels_text], dtype=np.int64)
    return features, labels, mapping


def load_dataset_archive(
    archive_path: str | Path,
    spec: DatasetSpec,
) -> ExternalDataset:
    """Verify and parse one archive according to its declared registry spec."""
    path = Path(archive_path)
    member = _read_member(path, spec)
    try:
        if spec.dataset_id == "uci-wdbc":
            features, labels, mapping = _parse_wdbc(member)
        elif spec.dataset_id == "uci-banknote":
            features, labels, mapping = _parse_banknote(member)
        elif spec.dataset_id == "uci-dry-bean":
            features, labels, mapping = _parse_arff(member)
        else:
            raise ValueError(f"no parser is registered for {spec.dataset_id!r}")
    except (UnicodeError, KeyError, IndexError, TypeError, ValueError) as exc:
        if str(exc).startswith("no parser is registered"):
            raise
        raise ValueError(f"{spec.dataset_id} archive member is malformed: {exc}") from exc
    if features.shape != (spec.n_rows, spec.n_features):
        raise ValueError(
            f"{spec.dataset_id} shape mismatch: got {features.shape}, "
            f"expected {(spec.n_rows, spec.n_features)}"
        )
    if labels.shape != (spec.n_rows,):
        raise ValueError(f"{spec.dataset_id} label shape mismatch")
    if not np.all(np.isfinite(features)):
        raise ValueError(f"{spec.dataset_id} contains non-finite features")
    if not np.array_equal(np.unique(labels), np.arange(spec.n_classes)):
        raise ValueError(f"{spec.dataset_id} class encoding is not contiguous")
    return ExternalDataset(
        spec=spec,
        features=features,
        labels=labels,
        label_mapping=mapping,
        archive_sha256=sha256_file(path),
        member_sha256=hashlib.sha256(member).hexdigest(),
    )


def fetch_external_dataset(
    dataset_id: str,
    *,
    cache_dir: str | Path,
) -> ExternalDataset:
    """Download-if-needed, verify, and parse a registered external dataset."""
    spec = get_dataset_spec(dataset_id)
    archive = _download_archive(spec, Path(cache_dir))
    return load_dataset_archive(archive, spec)


__all__ = [
    "ExternalDataset",
    "fetch_external_dataset",
    "load_dataset_archive",
]
