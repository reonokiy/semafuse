from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any
import time

from .model import HarnessError, Manifest
from .protocol import sha256_file


Downloader = Callable[..., str]


def _split_for_filename(filename: str) -> str:
    name = Path(filename).name
    for split in (
        "Accurate_Retrieval",
        "Conflict_Resolution",
        "Long_Range_Understanding",
        "Test_Time_Learning",
    ):
        if name.startswith(f"{split}-"):
            return split
    raise HarnessError(f"cannot derive split from manifest file {filename!r}")


def download_and_verify_files(
    manifest: Manifest,
    cache_dir: str | Path,
    *,
    splits: Iterable[str] | None = None,
    downloader: Downloader | None = None,
) -> dict[str, Any]:
    """Download pinned Hub files and verify size plus LFS SHA-256."""

    selected_splits = set(splits or manifest.raw["dataset"]["split_rows"])
    known_splits = set(manifest.raw["dataset"]["split_rows"])
    unknown_splits = selected_splits - known_splits
    if unknown_splits:
        raise HarnessError(f"unknown dataset splits: {sorted(unknown_splits)}")

    if downloader is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:  # pragma: no cover - exercised by real runs
            raise HarnessError("data verification requires huggingface-hub") from error
        downloader = hf_hub_download

    started = time.perf_counter()
    files: list[dict[str, Any]] = []
    for filename, expected in sorted(manifest.raw["dataset"]["files"].items()):
        split = _split_for_filename(filename)
        if split not in selected_splits:
            continue
        file_started = time.perf_counter()
        downloaded = Path(
            downloader(
                repo_id=manifest.dataset_repo_id,
                filename=filename,
                repo_type="dataset",
                revision=manifest.dataset_revision,
                cache_dir=str(Path(cache_dir).resolve()),
            )
        )
        actual_bytes = downloaded.stat().st_size
        actual_sha256 = sha256_file(downloaded)
        expected_bytes = int(expected["bytes"])
        expected_sha256 = str(expected["lfs_sha256"])
        if actual_bytes != expected_bytes:
            raise HarnessError(
                f"size mismatch for {filename}: expected {expected_bytes}, got {actual_bytes}"
            )
        if actual_sha256 != expected_sha256:
            raise HarnessError(
                f"SHA-256 mismatch for {filename}: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
        files.append(
            {
                "filename": filename,
                "split": split,
                "bytes": actual_bytes,
                "sha256": actual_sha256,
                "verified": True,
                "elapsed_seconds": time.perf_counter() - file_started,
            }
        )

    if not files:
        raise HarnessError("no manifest files selected for verification")
    return {
        "repo_id": manifest.dataset_repo_id,
        "revision": manifest.dataset_revision,
        "license": manifest.raw["dataset"]["license"],
        "selected_splits": sorted(selected_splits),
        "files": files,
        "all_verified": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
