"""PDF request encoding, editable transfer and checksum verification."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from urllib.parse import quote

from publication.zenodo_admission import _file_record, _integer_id
from publication.zenodo_errors import ZenodoError
from publication.zenodo_models import ZenodoDeposition, ZenodoFile
from publication.zenodo_operations import _ZenodoOperations


def _pdf_path(value: str | Path) -> Path:
    """Resolve an existing regular PDF path for upload or checksum work."""
    path = Path(value)
    if not path.is_file() or path.suffix.casefold() != ".pdf":
        raise ZenodoError(f"PDF file does not exist or is not a PDF: {path}")
    return path


def _safe_upload_filename(filename: str) -> str:
    """Reject multipart names that could break the Content-Disposition header."""
    if (
        not filename
        or Path(filename).name != filename
        or any(character in filename for character in '\"\r\n')
        or len(filename) > 255
    ):
        raise ZenodoError("PDF filename is not safe for multipart upload")
    return filename


def _multipart_body(field_name: str, filename: str, content: bytes) -> tuple[bytes, str]:
    safe_filename = _safe_upload_filename(filename)
    boundary = f"----active-fedference-{uuid.uuid4().hex}"
    marker = boundary.encode("ascii")
    chunks = [
        b"--" + marker + b"\r\n",
        f'Content-Disposition: form-data; name="name"\r\n\r\n{safe_filename}\r\n'.encode("utf-8"),
        b"--" + marker + b"\r\n",
        (
            f'Content-Disposition: form-data; name="{field_name}"; filename="{safe_filename}"\r\n'
            "Content-Type: application/pdf\r\n\r\n"
        ).encode("utf-8"),
        content,
        b"\r\n--" + marker + b"--\r\n",
    ]
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


class _ZenodoFileOperations(_ZenodoOperations):
    """Editable PDF transfer and exact local/server file verification."""

    def delete_file(self, deposition_id: int, file_id: str) -> None:
        """Delete one file from an editable deposition."""
        self._editable_deposition(deposition_id)
        if not isinstance(file_id, str) or not file_id.strip() or "/" in file_id:
            raise ZenodoError("Zenodo file id is unsafe")
        self._request(
            "DELETE",
            f"deposit/depositions/{_integer_id(deposition_id)}/files/{quote(file_id, safe='')}",
        )

    def upload_pdf(
        self,
        deposition_id: int,
        pdf_path: str | Path,
        *,
        replace_existing: bool = False,
    ) -> ZenodoFile:
        """Upload one PDF, returning the server checksum and file identity."""
        path = _pdf_path(pdf_path)
        content = path.read_bytes()
        expected_md5 = hashlib.md5(content).hexdigest()  # noqa: S324 - Zenodo's file API uses MD5
        deposition = self._editable_deposition(deposition_id)
        existing = [file for file in deposition.files if file.filename == path.name]
        if existing:
            if len(existing) == 1 and existing[0].filesize == len(content) and existing[0].checksum in {
                expected_md5,
                f"md5:{expected_md5}",
            }:
                return existing[0]
            if not replace_existing:
                raise ZenodoError(
                    f"Zenodo deposition already contains a different file named {path.name!r}; "
                    "inspect or replace the draft explicitly"
                )
            for record in existing:
                self.delete_file(deposition.id, record.id)
        body, content_type = _multipart_body("file", path.name, content)
        payload = self._request(
            "POST",
            f"deposit/depositions/{_integer_id(deposition_id)}/files",
            body=body,
            content_type=content_type,
        )
        return _file_record(payload)

    @staticmethod
    def _verify_pdf_record(
        deposition: ZenodoDeposition,
        pdf_path: str | Path,
        *,
        remote_filename: str | None,
        require_exact_file_set: bool,
    ) -> ZenodoFile:
        """Verify local bytes against one already-fetched deposition."""
        path = _pdf_path(pdf_path)
        expected_filename = _safe_upload_filename(remote_filename or path.name)
        if Path(expected_filename).suffix.casefold() != ".pdf":
            raise ZenodoError("verified Zenodo filename must use a .pdf extension")
        expected_md5 = hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 - Zenodo's file API uses MD5
        matches = [file for file in deposition.files if file.filename == expected_filename]
        if len(matches) != 1:
            raise ZenodoError(f"Zenodo deposition has {len(matches)} files named {expected_filename!r}")
        if require_exact_file_set and len(deposition.files) != 1:
            raise ZenodoError(
                "Zenodo deposition must contain exactly the verified PDF before publication"
            )
        record = matches[0]
        checksums = {expected_md5, f"md5:{expected_md5}"}
        if record.filesize != path.stat().st_size or record.checksum not in checksums:
            raise ZenodoError(f"Zenodo PDF checksum or size mismatch for {expected_filename}")
        return record

    def verify_pdf(
        self,
        deposition_id: int,
        pdf_path: str | Path,
        *,
        remote_filename: str | None = None,
        require_exact_file_set: bool = False,
    ) -> ZenodoFile:
        """Verify PDF bytes, optionally against a distinct server-side filename.

        A repository may use an informative copy name while Zenodo retains the
        canonical manuscript filename.  The alternate name is explicit so a
        checksum cannot silently match an unrelated deposition file.
        """
        path = _pdf_path(pdf_path)
        deposition = self.get_deposition(deposition_id)
        return self._verify_pdf_record(
            deposition,
            path,
            remote_filename=remote_filename,
            require_exact_file_set=require_exact_file_set,
        )
