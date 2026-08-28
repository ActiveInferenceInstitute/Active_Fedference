from __future__ import annotations

from publication.identifiers import doi_url, manuscript_pdf_filename, normalize_doi
from publication.release_assets import (
    CHECKSUM_FILENAME,
    GitHubReleaseAsset,
    GitHubReleaseVerification,
    ReleaseAssetError,
    ReleaseAssetInput,
    ReleaseAssetRecord,
    StagedReleaseAssets,
    expected_release_asset_names,
    load_github_release_assets,
    stage_release_assets,
    verify_github_release_downloads,
)
from publication.web_package import (
    WebPackageValidation,
    mirror_web_figures,
    normalize_web_xrefs,
    sanitize_machine_paths,
    validate_web_package,
)
from publication.zenodo import (
    ZenodoClient,
    ZenodoDeposition,
    ZenodoError,
    ZenodoFile,
    ZenodoMetadataSnapshot,
    token_from_env_file,
    token_from_environment,
)

__all__ = [
    "CHECKSUM_FILENAME",
    "GitHubReleaseAsset",
    "GitHubReleaseVerification",
    "ReleaseAssetError",
    "ReleaseAssetInput",
    "ReleaseAssetRecord",
    "StagedReleaseAssets",
    "WebPackageValidation",
    "mirror_web_figures",
    "normalize_web_xrefs",
    "doi_url",
    "expected_release_asset_names",
    "load_github_release_assets",
    "manuscript_pdf_filename",
    "normalize_doi",
    "sanitize_machine_paths",
    "stage_release_assets",
    "ZenodoClient",
    "ZenodoDeposition",
    "ZenodoError",
    "ZenodoFile",
    "ZenodoMetadataSnapshot",
    "token_from_env_file",
    "token_from_environment",
    "validate_web_package",
    "verify_github_release_downloads",
]
