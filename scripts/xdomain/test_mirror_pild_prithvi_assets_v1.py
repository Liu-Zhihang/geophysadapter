#!/usr/bin/env python3
"""Contract tests for the PILD Sentinel-2 local asset mirror."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import build_pild_prithvi_temporal_cache_v1 as builder
import mirror_pild_prithvi_assets_v1 as mirror


PAYLOAD = (b"sentinel-2-test-payload-" * 4096) + b"end"


class RangeHandler(BaseHTTPRequestHandler):
    ranges: list[str | None] = []

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self.__class__.ranges.append(self.headers.get("Range"))
        if self.path == "/expired":
            self.send_response(403)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path not in {"/asset", "/flaky"}:
            self.send_error(404)
            return
        range_header = self.headers.get("Range")
        if range_header:
            start = int(range_header.removeprefix("bytes=").removesuffix("-"))
            if start >= len(PAYLOAD):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(PAYLOAD)}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = PAYLOAD[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
        else:
            body = PAYLOAD
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.path == "/flaky" and len(body) > 32768:
            self.wfile.write(body[:32768])
            self.wfile.flush()
            self.close_connection = True
        else:
            self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


@contextmanager
def range_server():
    RangeHandler.ranges = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with patch.dict(
            os.environ,
            {"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"},
        ):
            yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class FakeResolver:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.refreshes: list[bool] = []

    def href(self, _item_id: str, _asset_name: str, refresh: bool = False) -> str:
        self.refreshes.append(refresh)
        return f"{self.base_url}/asset" if refresh else f"{self.base_url}/expired"


class FixedResolver:
    def __init__(self, href: str):
        self.fixed_href = href

    def href(self, _item_id: str, _asset_name: str, refresh: bool = False) -> str:
        return self.fixed_href


class MirrorDownloadTest(unittest.TestCase):
    def test_transfer_resumes_from_existing_part(self) -> None:
        with tempfile.TemporaryDirectory() as directory, range_server() as base_url:
            part = Path(directory) / "asset.tif.part"
            offset = 12345
            part.write_bytes(PAYLOAD[:offset])
            total = mirror.transfer_once(
                f"{base_url}/asset", part, timeout=(2, 2), chunk_size=1024
            )
            self.assertEqual(total, len(PAYLOAD))
            self.assertEqual(part.read_bytes(), PAYLOAD)
            self.assertEqual(RangeHandler.ranges, [f"bytes={offset}-"])

    def test_download_refreshes_expired_signed_url_and_commits_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory, range_server() as base_url:
            root = Path(directory)
            destination = root / "assets/item/B02.tif"
            resolver = FakeResolver(base_url)
            spec = mirror.AssetSpec(
                item_id="item",
                item_datetime="2020-01-01T00:00:00+00:00",
                collection=mirror.COLLECTION,
                asset_name="B02",
                unsigned_href="https://example.invalid/B02.tif",
                media_type="image/tiff",
                roles=("data",),
            )
            with patch.object(mirror.time, "sleep", return_value=None):
                record, skipped = mirror.download_asset(
                    spec,
                    destination,
                    "assets/item/B02.tif",
                    None,
                    resolver,  # type: ignore[arg-type]
                    "a" * 64,
                    retries=2,
                    timeout=(2, 2),
                    chunk_size=1024,
                )
            self.assertFalse(skipped)
            self.assertEqual(resolver.refreshes, [False, True])
            self.assertEqual(destination.read_bytes(), PAYLOAD)
            self.assertFalse(destination.with_name("B02.tif.part").exists())
            self.assertEqual(record["content_length"], len(PAYLOAD))
            self.assertEqual(record["sha256"], hashlib.sha256(PAYLOAD).hexdigest())

    def test_download_reconnects_and_resumes_after_truncated_responses(self) -> None:
        with tempfile.TemporaryDirectory() as directory, range_server() as base_url:
            root = Path(directory)
            destination = root / "assets/item/B03.tif"
            spec = mirror.AssetSpec(
                item_id="item",
                item_datetime="2020-01-01T00:00:00+00:00",
                collection=mirror.COLLECTION,
                asset_name="B03",
                unsigned_href="https://example.invalid/B03.tif",
                media_type="image/tiff",
                roles=("data",),
            )
            with patch.object(mirror.time, "sleep", return_value=None):
                record, _ = mirror.download_asset(
                    spec,
                    destination,
                    "assets/item/B03.tif",
                    None,
                    FixedResolver(f"{base_url}/flaky"),  # type: ignore[arg-type]
                    "a" * 64,
                    retries=8,
                    timeout=(2, 2),
                    chunk_size=4096,
                )
            self.assertEqual(destination.read_bytes(), PAYLOAD)
            self.assertGreater(len(RangeHandler.ranges), 1)
            self.assertEqual(record["sha256"], hashlib.sha256(PAYLOAD).hexdigest())


class BuilderMirrorValidationTest(unittest.TestCase):
    def build_mirror(self, root: Path, item_id: str = "item") -> tuple[Path, str]:
        availability_hash = "b" * 64
        manifest = root / "asset_mirror_manifest_v1.jsonl"
        records = []
        for asset_name in mirror.ASSETS:
            relative = Path("assets") / item_id / f"{asset_name}.tif"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            content = f"{item_id}/{asset_name}".encode()
            path.write_bytes(content)
            records.append(
                {
                    "schema_version": 1,
                    "availability_sha256": availability_hash,
                    "collection": mirror.COLLECTION,
                    "item_id": item_id,
                    "item_datetime": "2020-01-01T00:00:00+00:00",
                    "asset_name": asset_name,
                    "unsigned_href": f"https://example.invalid/{asset_name}.tif",
                    "media_type": "image/tiff",
                    "roles": ["data"],
                    "local_path": relative.as_posix(),
                    "content_length": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "completed_at_utc": "2020-01-02T00:00:00+00:00",
                }
            )
        manifest.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
        marker = builder.asset_mirror_marker_path(manifest)
        marker.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "complete": True,
                    "availability_sha256": availability_hash,
                    "manifest_sha256": mirror.sha256_file(manifest),
                    "asset_names": list(mirror.ASSETS),
                    "item_ids": [item_id],
                }
            ),
            encoding="utf-8",
        )
        return manifest, availability_hash

    def test_builder_accepts_complete_byte_verified_item(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, availability_hash = self.build_mirror(Path(directory))
            items, marker, manifest_hash = builder.load_local_asset_items(
                manifest, {"item"}, availability_hash
            )
            self.assertEqual(set(items["item"].assets), set(mirror.ASSETS))
            self.assertEqual(marker, builder.asset_mirror_marker_path(manifest))
            self.assertEqual(manifest_hash, mirror.sha256_file(manifest))
            self.assertTrue(all(Path(asset.href).is_file() for asset in items["item"].assets.values()))

    def test_builder_rejects_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, availability_hash = self.build_mirror(root)
            (root / "assets/item/B11.tif").write_bytes(b"same-length-bad")
            with self.assertRaisesRegex(RuntimeError, "length mismatch|SHA256 mismatch"):
                builder.load_local_asset_items(manifest, {"item"}, availability_hash)

    def test_builder_rejects_incomplete_marker_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, availability_hash = self.build_mirror(Path(directory))
            with self.assertRaisesRegex(RuntimeError, "complete scope lacks required items"):
                builder.load_local_asset_items(manifest, {"other-item"}, availability_hash)

    def test_builder_rejects_item_asset_path_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, availability_hash = self.build_mirror(Path(directory))
            lines = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            lines[0]["local_path"] = "assets/item/B03.tif"
            manifest.write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in lines),
                encoding="utf-8",
            )
            marker = builder.asset_mirror_marker_path(manifest)
            marker_data = json.loads(marker.read_text(encoding="utf-8"))
            marker_data["manifest_sha256"] = mirror.sha256_file(manifest)
            marker.write_text(json.dumps(marker_data), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "non-canonical local path"):
                builder.load_local_asset_items(manifest, {"item"}, availability_hash)

    def test_builder_accepts_subsecond_datetime_serialization_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, availability_hash = self.build_mirror(root)
            lines = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            lines[0]["item_datetime"] = "2020-01-01T00:00:00.024000+00:00"
            manifest.write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in lines),
                encoding="utf-8",
            )
            marker = builder.asset_mirror_marker_path(manifest)
            marker_data = json.loads(marker.read_text(encoding="utf-8"))
            marker_data["manifest_sha256"] = mirror.sha256_file(manifest)
            marker.write_text(json.dumps(marker_data), encoding="utf-8")
            items, _, _ = builder.load_local_asset_items(
                manifest, {"item"}, availability_hash
            )
            self.assertEqual(items["item"].datetime.isoformat(), "2020-01-01T00:00:00+00:00")

    def test_builder_rejects_material_datetime_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, availability_hash = self.build_mirror(root)
            lines = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            lines[0]["item_datetime"] = "2020-01-01T00:00:02+00:00"
            manifest.write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in lines),
                encoding="utf-8",
            )
            marker = builder.asset_mirror_marker_path(manifest)
            marker_data = json.loads(marker.read_text(encoding="utf-8"))
            marker_data["manifest_sha256"] = mirror.sha256_file(manifest)
            marker.write_text(json.dumps(marker_data), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "inconsistent item metadata"):
                builder.load_local_asset_items(manifest, {"item"}, availability_hash)


if __name__ == "__main__":
    unittest.main()
