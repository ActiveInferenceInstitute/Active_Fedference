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
from fedference.external_data import _download_archive, load_dataset_archive


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
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)
