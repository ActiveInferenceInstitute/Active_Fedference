"""Typed, composable boundary adapter for a Zenodo deposition.

The adapter intentionally has no project-specific paths and never prints or
stores bearer tokens.  Callers supply metadata, a deposition id, and files;
the thin CLI in ``scripts/zenodo_release.py`` supplies the release workflow.
Publication remains an explicit action rather than an implicit side effect of
reserving a DOI or uploading a file."""

from __future__ import annotations

from publication.zenodo_admission import _RFC3339_TIMESTAMP_RE as _RFC3339_TIMESTAMP_RE
from publication.zenodo_admission import _deposition as _deposition
from publication.zenodo_admission import _doi_for_record as _doi_for_record
from publication.zenodo_admission import _file_record as _file_record
from publication.zenodo_admission import _integer_id as _integer_id
from publication.zenodo_admission import _optional_integer_id as _optional_integer_id
from publication.zenodo_admission import _optional_text as _optional_text
from publication.zenodo_admission import _optional_utc_timestamp as _optional_utc_timestamp
from publication.zenodo_admission import _required_doi as _required_doi
from publication.zenodo_auth import DEFAULT_TOKEN_ENV_NAMES as DEFAULT_TOKEN_ENV_NAMES
from publication.zenodo_auth import _parse_env_value as _parse_env_value
from publication.zenodo_auth import token_from_env_file as token_from_env_file
from publication.zenodo_auth import token_from_environment as token_from_environment
from publication.zenodo_edits import _ZenodoMetadataOperations
from publication.zenodo_errors import ZenodoError as ZenodoError
from publication.zenodo_errors import _ZenodoHTTPError as _ZenodoHTTPError
from publication.zenodo_files import _multipart_body as _multipart_body
from publication.zenodo_files import _pdf_path as _pdf_path
from publication.zenodo_files import _safe_upload_filename as _safe_upload_filename
from publication.zenodo_lineage import _file_set_identity as _file_set_identity
from publication.zenodo_lineage import (
    _require_current_separate_record_metadata as _require_current_separate_record_metadata,
)
from publication.zenodo_lineage import _require_linked_version as _require_linked_version
from publication.zenodo_lineage import _require_published_snapshot as _require_published_snapshot
from publication.zenodo_lineage import _require_same_record_line as _require_same_record_line
from publication.zenodo_metadata import _SERVER_OWNED_METADATA_FIELDS as _SERVER_OWNED_METADATA_FIELDS
from publication.zenodo_metadata import _comparable_metadata as _comparable_metadata
from publication.zenodo_metadata import _request_metadata_comparison as _request_metadata_comparison
from publication.zenodo_metadata import _require_metadata_match as _require_metadata_match
from publication.zenodo_models import ZenodoDeposition as ZenodoDeposition
from publication.zenodo_models import ZenodoFile as ZenodoFile
from publication.zenodo_models import ZenodoMetadataSnapshot as ZenodoMetadataSnapshot
from publication.zenodo_models import _ZenodoLinkedVersionShape as _ZenodoLinkedVersionShape
from publication.zenodo_publish import _ZenodoPublicationOperations
from publication.zenodo_response import _ALREADY_EXISTS_RESPONSE as _ALREADY_EXISTS_RESPONSE
from publication.zenodo_response import _HTTP_ERROR_BODY_LIMIT as _HTTP_ERROR_BODY_LIMIT
from publication.zenodo_response import _JSON_RESPONSE_BODY_LIMIT as _JSON_RESPONSE_BODY_LIMIT
from publication.zenodo_response import _http_error_without_response_body as _http_error_without_response_body
from publication.zenodo_response import _is_existing_draft_response as _is_existing_draft_response
from publication.zenodo_response import _json_contains_token_echo as _json_contains_token_echo
from publication.zenodo_response import (
    _json_response_without_response_body as _json_response_without_response_body,
)
from publication.zenodo_transport import DEFAULT_ZENODO_API as DEFAULT_ZENODO_API
from publication.zenodo_transport import _api_base as _api_base
from publication.zenodo_transport import _RejectRedirects as _RejectRedirects
from publication.zenodo_versions import _DEPOSITION_LIST_PAGE_SIZE as _DEPOSITION_LIST_PAGE_SIZE
from publication.zenodo_versions import _FULL_DEPOSITION_FIELDS as _FULL_DEPOSITION_FIELDS
from publication.zenodo_versions import _latest_draft_link as _latest_draft_link
from publication.zenodo_versions import _ZenodoVersionOperations


class ZenodoClient(
    _ZenodoVersionOperations,
    _ZenodoMetadataOperations,
    _ZenodoPublicationOperations,
):
    """Small standard-library Zenodo REST client for release boundaries."""


__all__ = [
    "DEFAULT_TOKEN_ENV_NAMES",
    "DEFAULT_ZENODO_API",
    "ZenodoClient",
    "ZenodoDeposition",
    "ZenodoError",
    "ZenodoFile",
    "ZenodoMetadataSnapshot",
    "token_from_environment",
    "token_from_env_file",
]
