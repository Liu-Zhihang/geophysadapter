#!/usr/bin/env python3
"""Contract tests for STAC-free PILD Planetary Blob mirror recovery."""

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
from urllib.parse import parse_qs, urlsplit
from unittest.mock import patch

import resume_pild_planetary_blob_assets_v1 as blob
import run_pild_planetary_blob_mirror_shards_v1 as runner


ITEM_ID = "S2A_MSIL2A_20161220T060242_R091_T43TCE_20210213T224856"
SAFE_NAME = "S2A_MSIL2A_20161220T060242_N0212_R091_T43TCE_20210213T224856.SAFE"
GRANULE = "L2A_T43TCE_A007810_20161220T060240"
PAYLOAD = (b"planetary-blob-resume-" * 8192) + b"end"


def listing_xml(
    entries: list[tuple[str, int, str]], next_marker: str = ""
) -> bytes:
    blobs = "".join(
        "<Blob><Name>{}</Name><Properties><Content-Length>{}</Content-Length>"
        "<Etag>{}</Etag><Content-MD5>md5</Content-MD5>"
        "</Properties></Blob>".format(name, length, etag)
        for name, length, etag in entries
    )
    return (
        f"<?xml version='1.0'?><EnumerationResults><Blobs>{blobs}</Blobs>"
        f"<NextMarker>{next_marker}</NextMarker></EnumerationResults>"
    ).encode()


def resolved_entries(item_id: str = ITEM_ID) -> list[blob.BlobEntry]:
    identity = blob.parse_item_id(item_id)
    prefix = blob.blob_date_prefix(identity)
    result = []
    for index, (asset, (resolution, band, size)) in enumerate(blob.ASSET_LAYOUT.items(), start=1):
        name = (
            f"{prefix}{SAFE_NAME}/GRANULE/{GRANULE}/IMG_DATA/{resolution}/"
            f"T{identity.tile}_{identity.sensing}_{band}_{size}.tif"
        )
        result.append(blob.BlobEntry(name, index * 100, f'"etag-{asset}"', None))
    return result


class FakeResponse:
    def __init__(self, status_code: int, content: bytes = b"", payload: dict | None = None):
        self.status_code = status_code
        self.content = content
        self._payload = payload
        self.reason = "test"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise blob.requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("missing JSON payload")
        return self._payload


class PassProvider:
    def __init__(self):
        self.force_flags: list[bool] = []

    def sign(self, unsigned_url: str, force_refresh: bool = False) -> str:
        self.force_flags.append(force_refresh)
        return blob.append_query(unsigned_url, "sv=test&sig=signature")


class RangeHandler(BaseHTTPRequestHandler):
    ranges: list[str | None] = []

    def do_GET(self) -> None:  # noqa: N802
        self.__class__.ranges.append(self.headers.get("Range"))
        range_header = self.headers.get("Range")
        if range_header:
            offset = int(range_header.removeprefix("bytes=").removesuffix("-"))
            body = PAYLOAD[offset:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {offset}-{len(PAYLOAD)-1}/{len(PAYLOAD)}")
        else:
            body = PAYLOAD
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
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
            yield f"http://127.0.0.1:{server.server_port}/asset"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class IdentityAndResolutionTest(unittest.TestCase):
    def test_item_prefix_and_exact_identity(self) -> None:
        identity = blob.parse_item_id(ITEM_ID)
        self.assertEqual(identity.tile, "43TCE")
        self.assertEqual(
            blob.blob_product_prefix(identity),
            "43/T/CE/2016/12/20/S2A_MSIL2A_20161220T060242_",
        )
        self.assertEqual(blob.validate_safe_identity(identity, SAFE_NAME), "0212")

    def test_identity_mismatch_is_fatal(self) -> None:
        identity = blob.parse_item_id(ITEM_ID)
        wrong = SAFE_NAME.replace("R091", "R092")
        with self.assertRaisesRegex(RuntimeError, "identity mismatch.*orbit"):
            blob.validate_safe_identity(identity, wrong)
        wrong = SAFE_NAME.replace("20210213T224856", "20210213T224857")
        with self.assertRaisesRegex(RuntimeError, "identity mismatch.*generation"):
            blob.validate_safe_identity(identity, wrong)

    def test_asset_resolution_uses_frozen_10m_20m_contract(self) -> None:
        specs = blob.resolve_blob_assets(blob.parse_item_id(ITEM_ID), resolved_entries())
        self.assertEqual({spec.asset_name for spec in specs}, set(blob.ASSETS))
        resolutions = {spec.asset_name: spec.resolution for spec in specs}
        self.assertEqual({resolutions[name] for name in ("B02", "B03", "B04")}, {"R10m"})
        self.assertEqual(
            {resolutions[name] for name in ("B8A", "B11", "B12", "SCL")},
            {"R20m"},
        )
        self.assertEqual({spec.granule_name for spec in specs}, {GRANULE})
        self.assertTrue(all(spec.safe_name == SAFE_NAME for spec in specs))

    def test_real_planetary_blob_path_extracts_complete_safe_name(self) -> None:
        real_path = (
            "43/T/CE/2016/12/20/"
            "S2A_MSIL2A_20161220T060242_N0212_R091_T43TCE_20210213T224856.SAFE/"
            "GRANULE/L2A_T43TCE_A007810_20161220T060240/IMG_DATA/R10m/"
            "T43TCE_20161220T060242_B02_10m.tif"
        )
        entries = resolved_entries()
        entries[0] = blob.BlobEntry(real_path, 257186290, '"real-etag"', None)
        specs = blob.resolve_blob_assets(blob.parse_item_id(ITEM_ID), entries)
        b02 = next(spec for spec in specs if spec.asset_name == "B02")
        self.assertEqual(b02.safe_name, SAFE_NAME)
        self.assertEqual(b02.granule_name, GRANULE)
        self.assertEqual(b02.blob_name, real_path)

    def test_missing_or_duplicate_required_asset_is_fatal(self) -> None:
        entries = resolved_entries()
        with self.assertRaisesRegex(RuntimeError, "expected one SCL"):
            blob.resolve_blob_assets(blob.parse_item_id(ITEM_ID), entries[:-1])
        duplicate = blob.BlobEntry(
            entries[0].name.replace(GRANULE, "L2A_T43TCE_A999999_20161220T060240"),
            entries[0].content_length,
            '"other-etag"',
            None,
        )
        with self.assertRaisesRegex(RuntimeError, "expected one B02"):
            blob.resolve_blob_assets(blob.parse_item_id(ITEM_ID), entries + [duplicate])


class ListingAndTokenTest(unittest.TestCase):
    def test_listing_xml_pagination(self) -> None:
        prefix = blob.blob_product_prefix(blob.parse_item_id(ITEM_ID))
        page1 = listing_xml([(f"{prefix}first", 10, '"e1"')], "next-marker")
        page2 = listing_xml([(f"{prefix}second", 20, '"e2"')])
        markers: list[str] = []

        def fake_get(url: str, **_kwargs: object) -> FakeResponse:
            marker = parse_qs(urlsplit(url).query).get("marker", [""])[0]
            markers.append(marker)
            return FakeResponse(200, page2 if marker else page1)

        provider = PassProvider()
        with patch.object(blob.requests, "get", side_effect=fake_get):
            entries = blob.list_blobs(prefix, provider, retries=1, timeout=(1, 1))
        self.assertEqual([entry.name for entry in entries], [f"{prefix}first", f"{prefix}second"])
        self.assertEqual(markers, ["", "next-marker"])

    def test_listing_403_forces_token_refresh(self) -> None:
        prefix = blob.blob_product_prefix(blob.parse_item_id(ITEM_ID))
        responses = [FakeResponse(403), FakeResponse(200, listing_xml([(f"{prefix}ok", 1, '"e"')]))]
        provider = PassProvider()
        with patch.object(blob.requests, "get", side_effect=responses), patch.object(
            blob.time, "sleep", return_value=None
        ):
            entries = blob.list_blobs(prefix, provider, retries=2, timeout=(1, 1))
        self.assertEqual(len(entries), 1)
        self.assertEqual(provider.force_flags, [False, True])

    def test_sas_provider_caches_and_force_refreshes(self) -> None:
        responses = [
            FakeResponse(
                200,
                payload={"token": "sv=one&sig=first", "msft:expiry": "2099-01-01T00:00:00Z"},
            ),
            FakeResponse(
                200,
                payload={"token": "sv=two&sig=second", "msft:expiry": "2099-01-01T00:00:00Z"},
            ),
        ]
        provider = blob.SASTokenProvider(retries=1)
        with patch.object(blob.requests, "get", side_effect=responses) as mocked:
            first = provider.sign("https://example.test/blob?x=1")
            cached = provider.sign("https://example.test/blob?x=1")
            refreshed = provider.sign("https://example.test/blob?x=1", force_refresh=True)
        self.assertEqual(first, cached)
        self.assertIn("x=1&sv=one&sig=first", first)
        self.assertIn("x=1&sv=two&sig=second", refreshed)
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(provider.refresh_count, 2)


class ResumeAndShardTest(unittest.TestCase):
    def test_existing_part_resumes_and_commits_with_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory, range_server() as href:
            root = Path(directory)
            destination = root / "assets" / ITEM_ID / "B02.tif"
            destination.parent.mkdir(parents=True)
            part = destination.with_name("B02.tif.part")
            offset = 32123
            part.write_bytes(PAYLOAD[:offset])
            spec = blob.BlobAssetSpec(
                item_id=ITEM_ID,
                safe_name=SAFE_NAME,
                granule_name=GRANULE,
                processing_baseline="0212",
                asset_name="B02",
                resolution="R10m",
                blob_name="test/B02.tif",
                unsigned_href=href,
                content_length=len(PAYLOAD),
                etag='"etag"',
                content_md5=None,
            )
            record, skipped = blob.download_asset(
                spec,
                destination,
                f"assets/{ITEM_ID}/B02.tif",
                None,
                PassProvider(),  # type: ignore[arg-type]
                "a" * 64,
                retries=2,
                timeout=(2, 2),
                chunk_size=4096,
            )
            self.assertFalse(skipped)
            self.assertEqual(RangeHandler.ranges, [f"bytes={offset}-"])
            self.assertEqual(destination.read_bytes(), PAYLOAD)
            self.assertFalse(part.exists())
            self.assertEqual(record["content_length"], len(PAYLOAD))
            self.assertEqual(record["sha256"], hashlib.sha256(PAYLOAD).hexdigest())
            self.assertFalse(record["stac_api_used"])

    def test_four_shards_are_disjoint_and_complete(self) -> None:
        items = [f"item-{index:03d}" for index in range(288)]
        partitions = runner.partition_item_ids(items, 4)
        self.assertEqual([len(values) for values in partitions], [72, 72, 72, 72])
        self.assertEqual(set().union(*map(set, partitions)), set(items))
        for left in range(4):
            for right in range(left + 1, 4):
                self.assertTrue(set(partitions[left]).isdisjoint(partitions[right]))

    def test_resolution_manifest_conflict_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resolution.jsonl"
            base = {
                "schema_version": 1,
                "availability_sha256": "a" * 64,
                "stac_api_used": False,
                "acquisition_identity_match": True,
                "item_id": ITEM_ID,
                "asset_name": "B02",
                "blob_name": "one",
            }
            other = {**base, "blob_name": "two"}
            path.write_text(
                json.dumps(base, sort_keys=True) + "\n" + json.dumps(other, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "conflicting Blob resolution"):
                blob.load_resolution_manifest(path, "a" * 64)


if __name__ == "__main__":
    unittest.main()
