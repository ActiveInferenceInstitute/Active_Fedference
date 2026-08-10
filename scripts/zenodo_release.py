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
    parser.add_argument("--deposition-id", type=int)
    parser.add_argument("--reserve", action="store_true", help="create a new unsubmitted DOI draft")
    parser.add_argument(
        "--update-metadata",
        action="store_true",
        help="replace editable metadata on an existing deposition from --metadata",
    )
    parser.add_argument("--upload", type=Path, help="upload this PDF to the selected deposition")
    parser.add_argument("--verify", type=Path, metavar="PDF", help="verify this uploaded PDF")
    parser.add_argument("--publish", action="store_true", help="publish the selected deposition")
    parser.add_argument(
        "--confirm-publish",
        action="store_true",
        help="required with --publish; publication cannot be undone",
    )
    args = parser.parse_args(argv)

    from publication.zenodo import ZenodoClient, ZenodoError, token_from_env_file, token_from_environment

    try:
        if args.reserve and args.deposition_id is not None:
            parser.error("--reserve and --deposition-id are mutually exclusive")
        if args.update_metadata and args.reserve:
            parser.error("--update-metadata requires an existing --deposition-id")
        if args.publish and not args.confirm_publish:
            parser.error("--publish requires --confirm-publish")
        if args.confirm_publish and not args.publish:
            parser.error("--confirm-publish requires --publish")
        if args.env_file is not None:
            token, token_name = token_from_env_file(args.env_file)
        else:
            token, token_name = token_from_environment()
        client = ZenodoClient(token)

        if args.reserve:
            metadata = _load_metadata(args.metadata)
            deposition = client.reserve_doi(metadata)
        else:
            if args.deposition_id is None:
                parser.error("one of --reserve or --deposition-id is required")
            deposition = client.get_deposition(args.deposition_id)

        if args.update_metadata:
            metadata = _load_metadata(args.metadata)
            # Zenodo owns the DOI after reservation; the repository-side
            # generated surface may contain it, but the editable API metadata
            # must not try to replace that server-owned identifier.
            metadata.pop("doi", None)
            metadata.setdefault("access_right", "open")
            deposition = client.update_metadata(deposition.id, metadata)

        if args.upload is not None:
            client.upload_pdf(deposition.id, args.upload)
            deposition = client.get_deposition(deposition.id)
        if args.verify is not None:
            client.verify_pdf(deposition.id, args.verify)
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
