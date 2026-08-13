#!/usr/bin/env python3
"""Reserve, upload, verify, or explicitly publish the Zenodo PDF deposition.

The default operations are reversible draft preparation steps.  Publishing is
irreversible and requires both ``--publish`` and ``--confirm-publish``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def _load_metadata(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Zenodo metadata must be a JSON object: {path}")
    return payload


def _summary(deposition: Any) -> dict[str, Any]:
    return {
        "id": deposition.id,
        "state": deposition.state,
        "doi": deposition.doi,
        "reserved_doi": deposition.reserved_doi,
        "html_url": deposition.html_url,
        "files": [
            {
                "filename": file.filename,
                "filesize": file.filesize,
                "checksum": file.checksum,
            }
            for file in deposition.files
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """Run one explicit Zenodo deposition operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, help="dotenv file containing a Zenodo token")
    parser.add_argument("--metadata", type=Path, default=Path(".zenodo.json"))
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="standalone checkout containing metadata and release files",
    )
    parser.add_argument("--deposition-id", type=int)
    parser.add_argument(
        "--new-version-of",
        type=int,
        metavar="DEPOSITION_ID",
        help="create or reuse the unpublished next version of a published deposition",
    )
    parser.add_argument("--reserve", action="store_true", help="create a new unsubmitted DOI draft")
    parser.add_argument(
        "--update-metadata",
        action="store_true",
        help="replace editable metadata on an existing deposition from --metadata",
    )
    parser.add_argument("--upload", type=Path, help="upload this PDF to the selected deposition")
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="replace a same-named inherited draft file before upload",
    )
    parser.add_argument("--verify", type=Path, metavar="PDF", help="verify this uploaded PDF")
    parser.add_argument(
        "--remote-filename",
        help="server-side filename to verify when it differs from the local PDF name",
    )
    parser.add_argument("--publish", action="store_true", help="publish the selected deposition")
    parser.add_argument(
        "--confirm-publish",
        action="store_true",
        help="required with --publish; publication cannot be undone",
    )
    args = parser.parse_args(argv)

    from project_paths import resolve_script_project_root
    from publication.zenodo import ZenodoClient, ZenodoError, token_from_env_file, token_from_environment

    try:
        root = resolve_script_project_root(_PROJECT_ROOT, args.project_root)
        env_file_path = args.env_file
        if env_file_path is not None and not env_file_path.is_absolute():
            env_file_path = root / env_file_path
        metadata_path = args.metadata if args.metadata.is_absolute() else root / args.metadata
        upload_path = args.upload
        if upload_path is not None and not upload_path.is_absolute():
            upload_path = root / upload_path
        verify_path = args.verify
        if verify_path is not None and not verify_path.is_absolute():
            verify_path = root / verify_path
        if args.reserve and (args.deposition_id is not None or args.new_version_of is not None):
            parser.error("--reserve cannot be combined with --deposition-id or --new-version-of")
        if args.deposition_id is not None and args.new_version_of is not None:
            parser.error("--deposition-id and --new-version-of are mutually exclusive")
        if args.update_metadata and args.reserve:
            parser.error("--update-metadata requires an existing --deposition-id")
        if args.replace_existing and upload_path is None:
            parser.error("--replace-existing requires --upload")
        if args.publish and not args.confirm_publish:
            parser.error("--publish requires --confirm-publish")
        if args.publish and verify_path is None:
            parser.error("--publish requires --verify PDF so the uploaded bytes are checked first")
        if args.confirm_publish and not args.publish:
            parser.error("--confirm-publish requires --publish")
        if env_file_path is not None:
            token, token_name = token_from_env_file(env_file_path)
        else:
            token, token_name = token_from_environment()
        client = ZenodoClient(token)

        if args.reserve:
            metadata = _load_metadata(metadata_path)
            deposition = client.reserve_doi(metadata)
        elif args.new_version_of is not None:
            deposition = client.new_version(args.new_version_of)
        else:
            if args.deposition_id is None:
                parser.error("one of --reserve, --new-version-of, or --deposition-id is required")
            deposition = client.get_deposition(args.deposition_id)

        if args.update_metadata:
            metadata = _load_metadata(metadata_path)
            # Zenodo owns the DOI after reservation; the repository-side
            # generated surface may contain it, but the editable API metadata
            # must not try to replace that server-owned identifier.
            metadata.pop("doi", None)
            metadata.setdefault("access_right", "open")
            deposition = client.update_metadata(deposition.id, metadata)

        if upload_path is not None:
            client.upload_pdf(
                deposition.id,
                upload_path,
                replace_existing=args.replace_existing,
            )
            deposition = client.get_deposition(deposition.id)
        if verify_path is not None:
            client.verify_pdf(
                deposition.id,
                verify_path,
                remote_filename=args.remote_filename,
            )
            deposition = client.get_deposition(deposition.id)
        if args.publish:
            deposition = client.publish(deposition.id)

        result = _summary(deposition)
        result["token_source"] = token_name
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, ZenodoError) as exc:
        print(f"Zenodo release operation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
