#!/usr/bin/env python3
"""Download pinned Hugging Face dataset files with resumable parallel HTTP."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile


@dataclass(frozen=True)
class FileSpec:
    path: str
    size: int
    sha256: str


def sha256sum(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def list_specs(repo_id: str, revision: str, prefixes: tuple[str, ...]) -> list[FileSpec]:
    specs: list[FileSpec] = []
    entries = HfApi().list_repo_tree(
        repo_id,
        repo_type="dataset",
        revision=revision,
        recursive=True,
        expand=True,
    )
    for entry in entries:
        if not isinstance(entry, RepoFile):
            continue
        if prefixes and not any(
            entry.path == prefix or entry.path.startswith(prefix.rstrip("/") + "/")
            for prefix in prefixes
        ):
            continue
        if entry.lfs is None:
            raise RuntimeError(f"Expected an LFS checksum for {entry.path}")
        specs.append(FileSpec(entry.path, int(entry.size), entry.lfs.sha256))
    if not specs:
        raise RuntimeError(f"No files matched prefixes: {prefixes}")
    return sorted(specs, key=lambda item: item.path)


def resolve_url(repo_id: str, revision: str, path: str, endpoint: str) -> str:
    encoded = "/".join(quote(part, safe="") for part in path.split("/"))
    return f"{endpoint.rstrip('/')}/datasets/{repo_id}/resolve/{revision}/{encoded}?download=true"


def download_one(
    spec: FileSpec,
    repo_id: str,
    revision: str,
    endpoint: str,
    out_dir: Path,
    attempts: int,
    verify_existing: bool,
) -> dict[str, object]:
    target = out_dir / spec.path
    part = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.is_file() and target.stat().st_size == spec.size:
        if not verify_existing or sha256sum(target) == spec.sha256:
            return {"path": spec.path, "status": "existing", "bytes": spec.size}

    if target.exists():
        target.unlink()
    if part.exists() and part.stat().st_size > spec.size:
        part.unlink()

    url = resolve_url(repo_id, revision, spec.path, endpoint)
    last_error = ""
    for attempt in range(1, attempts + 1):
        command = [
            "curl",
            "-fL",
            "-C",
            "-",
            "--connect-timeout",
            "30",
            "--speed-time",
            "180",
            "--speed-limit",
            "1024",
            "--retry",
            "6",
            "--retry-all-errors",
            "--retry-delay",
            "5",
            "--silent",
            "--show-error",
            url,
            "-o",
            str(part),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode == 0 and part.is_file() and part.stat().st_size == spec.size:
            actual = sha256sum(part)
            if actual == spec.sha256:
                part.replace(target)
                return {"path": spec.path, "status": "downloaded", "bytes": spec.size}
            last_error = f"sha256 mismatch: {actual} != {spec.sha256}"
            part.unlink(missing_ok=True)
        else:
            actual_size = part.stat().st_size if part.exists() else 0
            last_error = (
                f"curl={completed.returncode}, bytes={actual_size}/{spec.size}, "
                f"stderr={completed.stderr[-500:].strip()}"
            )
            if completed.returncode == 33:
                part.unlink(missing_ok=True)
        if attempt < attempts:
            time.sleep(min(30, 3 * attempt))
    raise RuntimeError(f"{spec.path}: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", action="append", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()

    specs = list_specs(args.repo_id, args.revision, tuple(args.prefix))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "revision": args.revision,
                "endpoint": args.endpoint,
                "total_files": len(specs),
                "total_bytes": sum(spec.size for spec in specs),
                "files": [asdict(spec) for spec in specs],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"[INFO] {len(specs)} files, {sum(item.size for item in specs) / 2**30:.2f} GiB, "
        f"workers={args.workers}",
        flush=True,
    )
    failures: list[str] = []
    completed_bytes = 0
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_one,
                spec,
                args.repo_id,
                args.revision,
                args.endpoint,
                args.out_dir,
                args.attempts,
                args.verify_existing,
            ): spec
            for spec in specs
        }
        for future in concurrent.futures.as_completed(futures):
            spec = futures[future]
            try:
                result = future.result()
            except Exception as error:  # keep other independent downloads alive
                failures.append(f"{spec.path}: {error}")
                print(f"[FAIL] {spec.path}: {error}", file=sys.stderr, flush=True)
                continue
            with lock:
                completed_bytes += int(result["bytes"])
                print(
                    f"[{result['status'].upper()}] {result['path']} "
                    f"({completed_bytes / 2**30:.2f}/{sum(item.size for item in specs) / 2**30:.2f} GiB)",
                    flush=True,
                )
    if failures:
        failure_path = args.manifest.with_suffix(".failures.json")
        failure_path.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        return 1
    args.manifest.with_suffix(".complete").touch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
