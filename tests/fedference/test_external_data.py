"""Pinned external archive parsing without network test doubles."""

from __future__ import annotations

import zipfile
from dataclasses import replace
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import numpy as np
import pytest

from fedference.evidence import DatasetSpec, sha256_file
from fedference.external_data import (
    ExternalDataset,
    _download_archive,
    _parse_arff,
    _parse_wdbc,
    _read_member,
    load_dataset_archive,
)


def _banknote_archive(tmp_path):
    archive = tmp_path / "banknote.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(
            "data_banknote_authentication.txt",
            "1.0,2.0,3.0,4.0,0\n2.0,3.0,4.0,5.0,1\n3.0,4.0,5.0,6.0,0\n",
        )
    spec = DatasetSpec(
        dataset_id="uci-banknote",
        name="Banknote fixture",
        source_url="https://example.test/banknote.zip",
        doi="10.0000/example",
        license="CC BY 4.0",
        archive_sha256=sha256_file(archive),
        archive_member="data_banknote_authentication.txt",
        file_format="csv",
        n_rows=3,
        n_features=4,
        n_classes=2,
        has_missing_values=False,
        preprocessing=("parse",),
        schema=("features: four float64 values", "class: integer"),
        split_policy="seeded holdout with receipt-bound split hash",
    )
    return archive, spec


def test_verified_archive_parses_to_contiguous_labels(tmp_path) -> None:
    archive, spec = _banknote_archive(tmp_path)
    dataset = load_dataset_archive(archive, spec)
    assert dataset.features.shape == (3, 4)
    assert dataset.labels.tolist() == [0, 1, 0]
    assert dataset.label_mapping == (("0", 0), ("1", 1))
    assert dataset.archive_sha256 == spec.archive_sha256
    assert len(dataset.member_sha256) == 64
    assert np.all(np.isfinite(dataset.features))
    with pytest.raises(ValueError, match="read-only"):
        dataset.features[0, 0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        dataset.labels[0] = 1
    with pytest.raises(ValueError, match="declared dataset shape"):
        replace(dataset, features=np.zeros((2, 4)))
    with pytest.raises(ValueError, match="label_mapping"):
        replace(dataset, label_mapping=(("0", 0), ("duplicate", 0)))
    with pytest.raises(ValueError, match="integer-valued"):
        replace(dataset, labels=np.asarray([0.0, 0.5, 1.0]))
    with pytest.raises(ValueError, match="real numeric"):
        replace(dataset, features=dataset.features.astype(str))
    with pytest.raises(ValueError, match="real integer-valued"):
        replace(dataset, labels=np.asarray([False, True, False]))


def test_archive_digest_and_shape_mismatches_fail_closed(tmp_path) -> None:
    archive, spec = _banknote_archive(tmp_path)
    with pytest.raises(ValueError, match="digest mismatch"):
        load_dataset_archive(archive, replace(spec, archive_sha256="0" * 64))
    with pytest.raises(ValueError, match="shape mismatch"):
        load_dataset_archive(archive, replace(spec, n_rows=4))


@pytest.mark.parametrize(
    "bad_row",
    (
        "1.0,2.0,3.0,0\n",
        "1.0,2.0,3.0,4.0,0.5\n",
        "1.0,2.0,3.0,4.0,2\n",
        "1.0,2.0,nope,4.0,0\n",
    ),
)
def test_banknote_parser_rejects_malformed_rows(tmp_path, bad_row) -> None:
    archive = tmp_path / "bad-banknote.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("data_banknote_authentication.txt", bad_row)
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    _, base_spec = _banknote_archive(base_dir)
    spec = replace(
        base_spec,
        archive_sha256=sha256_file(archive),
        n_rows=1,
    )
    with pytest.raises(ValueError, match="malformed"):
        load_dataset_archive(archive, spec)


def test_archive_download_uses_real_http_and_cleans_digest_failures(tmp_path) -> None:
    served = tmp_path / "served"
    served.mkdir()
    archive, base_spec = _banknote_archive(served)
    handler = partial(SimpleHTTPRequestHandler, directory=str(served))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        source_url = f"http://127.0.0.1:{server.server_port}/{archive.name}"
        spec = replace(base_spec, source_url=source_url)
        cache = tmp_path / "cache"
        downloaded = _download_archive(spec, cache)
        assert sha256_file(downloaded) == spec.archive_sha256
        assert _download_archive(spec, cache) == downloaded

        wrong_digest = replace(spec, archive_sha256="0" * 64)
        with pytest.raises(ValueError, match="downloaded archive digest mismatch"):
            _download_archive(wrong_digest, cache)
        assert not list(cache.glob("*.tmp"))
        downloaded.write_bytes(b"tampered")
        with pytest.raises(ValueError, match="cached archive digest mismatch"):
            _download_archive(spec, cache)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


def test_wdbc_parser_and_archive_reader_cover_diagnosis_mapping(tmp_path) -> None:
    row_b = ",".join(["id-b", "B", *(["1.0"] * 30)])
    row_m = ",".join(["id-m", "M", *(["2.0"] * 30)])
    member = (row_b + "\n" + row_m + "\n").encode()
    features, labels, mapping = _parse_wdbc(member)
    assert features.shape == (2, 30)
    assert labels.tolist() == [0, 1]
    assert mapping == (("B", 0), ("M", 1))

    archive = tmp_path / "wdbc.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("wdbc.data", member)
    spec = DatasetSpec(
        dataset_id="uci-wdbc",
        name="WDBC fixture",
        source_url="https://example.test/wdbc.zip",
        doi="10.0000/wdbc",
        license="CC BY 4.0",
        archive_sha256=sha256_file(archive),
        archive_member="wdbc.data",
        file_format="csv",
        n_rows=2,
        n_features=30,
        n_classes=2,
        has_missing_values=False,
        preprocessing=("parse",),
        schema=("features: thirty values", "class: B/M"),
        split_policy="seeded holdout",
    )
    assert _read_member(archive, spec) == member
    dataset = load_dataset_archive(archive, spec)
    assert isinstance(dataset, ExternalDataset)
    assert dataset.labels.tolist() == [0, 1]


def test_arff_parser_maps_sorted_nominal_labels_and_rejects_empty_or_ragged_data() -> None:
    data = b"""% comment
@RELATION fixture
@ATTRIBUTE x NUMERIC
@ATTRIBUTE y NUMERIC
@ATTRIBUTE class {z,a}
@DATA
1.0,2.0,z
3.0,4.0,a
"""
    features, labels, mapping = _parse_arff(data)
    np.testing.assert_allclose(features, [[1.0, 2.0], [3.0, 4.0]])
    assert labels.tolist() == [1, 0]
    assert mapping == (("a", 0), ("z", 1))

    with pytest.raises(ValueError, match="no data rows"):
        _parse_arff(b"@RELATION empty\n@DATA\n")
    with pytest.raises(ValueError, match="consistent"):
        _parse_arff(b"@DATA\n1,yes\n2,3,no\n")


def test_archive_reader_rejects_invalid_zip_and_missing_member(tmp_path) -> None:
    archive = tmp_path / "invalid.zip"
    archive.write_bytes(b"not a zip")
    valid_dir = tmp_path / "valid"
    valid_dir.mkdir()
    _, spec = _banknote_archive(valid_dir)
    spec = replace(spec, archive_sha256=sha256_file(archive))
    with pytest.raises(ValueError, match="invalid ZIP"):
        _read_member(archive, spec)

    missing = tmp_path / "missing.zip"
    with zipfile.ZipFile(missing, "w") as bundle:
        bundle.writestr("other.txt", "content")
    missing_spec = replace(spec, archive_sha256=sha256_file(missing))
    with pytest.raises(ValueError, match="is missing"):
        _read_member(missing, missing_spec)


def test_unknown_external_dataset_parser_is_explicitly_rejected(tmp_path) -> None:
    archive, base = _banknote_archive(tmp_path)
    unknown = replace(base, dataset_id="unknown-fixture")
    with pytest.raises(ValueError, match="no parser is registered"):
        load_dataset_archive(archive, unknown)
