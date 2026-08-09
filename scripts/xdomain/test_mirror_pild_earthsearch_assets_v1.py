#!/usr/bin/env python3
"""Contract tests for the independent PILD Earth Search asset mirror."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mirror_pild_earthsearch_assets_v1 as mirror
import run_pild_earthsearch_mirror_shards_v1 as runner


PC_ID = "S2A_MSIL2A_20180815T131241_R138_T23KNT_20201011T020243"
ES_ID = "S2A_23KNT_20180815_0_L2A"


def earthsearch_item(
    *,
    item_id: str = ES_ID,
    platform: str = "sentinel-2a",
    tile: str = "23KNT",
    sensing: str = "20180815T131241",
    orbit: str = "138",
) -> dict:
    assets = {
        earth_name: {
            "href": f"https://sentinel-cogs.example/{item_id}/{output_name}.tif",
            "type": "image/tiff; application=geotiff; profile=cloud-optimized",
            "roles": ["data"],
        }
        for output_name, earth_name in mirror.ASSET_MAP.items()
    }
    return {
        "id": item_id,
        "properties": {
            "platform": platform,
            "datetime": "2018-08-15T13:12:42.456000Z",
            "grid:code": f"MGRS-{tile}",
            "s2:product_uri": (
                f"S2A_MSIL2A_{sensing}_N0001_R{orbit}_T{tile}_20200921T125011.SAFE"
            ),
        },
        "assets": assets,
    }


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload
        self.reason = "test"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise mirror.requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("no payload")
        return self._payload


class IdentityTest(unittest.TestCase):
    def test_pc_id_parsing_and_occurrence_candidates(self) -> None:
        identity = mirror.parse_pc_item_id(PC_ID)
        self.assertEqual(identity.satellite, "S2A")
        self.assertEqual(identity.platform, "sentinel-2a")
        self.assertEqual(identity.sensing, "20180815T131241")
        self.assertEqual(identity.orbit, "138")
        self.assertEqual(identity.tile, "23KNT")
        self.assertEqual(
            mirror.candidate_earthsearch_ids(PC_ID),
            [f"S2A_23KNT_20180815_{index}_L2A" for index in range(4)],
        )

    def test_band_mapping_is_exact_and_complete(self) -> None:
        self.assertEqual(
            mirror.ASSET_MAP,
            {
                "B02": "blue",
                "B03": "green",
                "B04": "red",
                "B8A": "nir08",
                "B11": "swir16",
                "B12": "swir22",
                "SCL": "scl",
            },
        )
        item = mirror.validate_earthsearch_item(PC_ID, earthsearch_item())
        specs = mirror.item_to_specs(item)
        self.assertEqual({spec.asset_name for spec in specs}, set(mirror.ASSETS))
        self.assertEqual(
            {spec.asset_name: spec.earthsearch_asset_name for spec in specs},
            mirror.ASSET_MAP,
        )
        self.assertTrue(all(spec.acquisition_identity_match for spec in specs))

    def test_identity_mismatch_is_fatal(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "identity mismatch.*property_tile"):
            mirror.validate_earthsearch_item(PC_ID, earthsearch_item(tile="24KAA"))
        with self.assertRaisesRegex(RuntimeError, "identity mismatch.*product_orbit"):
            mirror.validate_earthsearch_item(PC_ID, earthsearch_item(orbit="139"))
        with self.assertRaisesRegex(RuntimeError, "identity mismatch.*platform"):
            mirror.validate_earthsearch_item(PC_ID, earthsearch_item(platform="sentinel-2b"))

    def test_resolver_tries_bounded_occurrences(self) -> None:
        calls: list[str] = []

        def fake_get(url: str, **_kwargs: object) -> FakeResponse:
            calls.append(url)
            if "_0_L2A" in url:
                return FakeResponse(404)
            if "_1_L2A" in url:
                payload = earthsearch_item(item_id="S2A_23KNT_20180815_1_L2A")
                return FakeResponse(200, payload)
            raise AssertionError("resolver exceeded first matching occurrence")

        with patch.object(mirror.requests, "get", side_effect=fake_get):
            result = mirror.fetch_earthsearch_item(PC_ID, retries=1, timeout=(1, 1))
        self.assertEqual(result.earthsearch_item_id, "S2A_23KNT_20180815_1_L2A")
        self.assertEqual(len(calls), 2)

    def test_resolver_skips_mismatched_occurrence_then_matches(self) -> None:
        calls: list[str] = []

        def fake_get(url: str, **_kwargs: object) -> FakeResponse:
            calls.append(url)
            if "_0_L2A" in url:
                return FakeResponse(200, earthsearch_item(orbit="139"))
            if "_1_L2A" in url:
                return FakeResponse(
                    200,
                    earthsearch_item(item_id="S2A_23KNT_20180815_1_L2A"),
                )
            raise AssertionError("resolver exceeded first identity match")

        with patch.object(mirror.requests, "get", side_effect=fake_get):
            result = mirror.fetch_earthsearch_item(PC_ID, retries=1, timeout=(1, 1))
        self.assertEqual(result.earthsearch_item_id, "S2A_23KNT_20180815_1_L2A")
        self.assertEqual(len(calls), 2)


class ShardingAndManifestTest(unittest.TestCase):
    def test_shard_partitions_are_disjoint_and_complete(self) -> None:
        items = [f"item-{index}" for index in range(17)]
        partitions = runner.partition_item_ids(items, 4)
        self.assertEqual(set().union(*map(set, partitions)), set(items))
        self.assertEqual(sum(map(len, partitions)), len(items))
        for left in range(len(partitions)):
            for right in range(left + 1, len(partitions)):
                self.assertTrue(set(partitions[left]).isdisjoint(partitions[right]))

    def test_manifest_conflict_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            base = {
                "schema_version": 1,
                "mirror_source": "earth-search",
                "availability_sha256": "a" * 64,
                "original_item_id": PC_ID,
                "earthsearch_item_id": ES_ID,
                "earthsearch_product_uri": (
                    "S2A_MSIL2A_20180815T131241_N0001_R138_T23KNT_"
                    "20200921T125011.SAFE"
                ),
                "acquisition_identity_match": True,
                "asset_name": "B02",
                "earthsearch_asset_name": "blue",
                "public_href": f"https://sentinel-cogs.example/{ES_ID}/B02.tif",
                "local_path": f"assets/{PC_ID}/B02.tif",
                "content_length": 10,
                "sha256": "b" * 64,
            }
            conflict = {**base, "sha256": "c" * 64}
            path.write_text(
                json.dumps(base, sort_keys=True) + "\n" + json.dumps(conflict, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "conflicting Earth Search manifest"):
                mirror.load_manifest(path, "a" * 64)

    def test_merge_conflict_is_fatal(self) -> None:
        key = (PC_ID, "B02")
        destination = {key: {"sha256": "a"}}
        with self.assertRaisesRegex(RuntimeError, "conflicting Earth Search shard records"):
            runner.merge_records(destination, {key: {"sha256": "b"}})

    def test_non_earthsearch_manifest_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "mirror_source": "planetary-computer",
                        "availability_sha256": "a" * 64,
                        "acquisition_identity_match": True,
                        "original_item_id": PC_ID,
                        "asset_name": "B02",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "non-Earth-Search"):
                mirror.load_manifest(path, "a" * 64)

    def test_valid_file_cannot_hide_resolver_manifest_conflict(self) -> None:
        item = mirror.validate_earthsearch_item(PC_ID, earthsearch_item())
        spec = mirror.item_to_specs(item)[0]
        relative = f"assets/{PC_ID}/B02.tif"
        record = {
            "mirror_source": "earth-search",
            "original_item_id": PC_ID,
            "earthsearch_item_id": "S2A_23KNT_20180815_1_L2A",
            "earthsearch_product_uri": spec.earthsearch_product_uri,
            "acquisition_identity_match": True,
            "asset_name": "B02",
            "earthsearch_asset_name": "blue",
            "public_href": spec.href,
            "local_path": relative,
        }
        with self.assertRaisesRegex(RuntimeError, "conflicts with resolved asset"):
            mirror.validate_prior_record(spec, record, relative)

    def test_planetary_computer_root_artifacts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "asset_mirror_manifest_v1.jsonl").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Planetary Computer mirror artifacts"):
                mirror.assert_independent_mirror_root(root)


if __name__ == "__main__":
    unittest.main()
