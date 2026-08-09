#!/usr/bin/env python3

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("audit_mechanism_aware_trigger_support_v1.py")
SPEC = importlib.util.spec_from_file_location("trigger_audit", MODULE_PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MOD)


class BytesResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, size=-1):
        return self.payload if size < 0 else self.payload[:size]


class TriggerAuditTests(unittest.TestCase):
    def test_candidate_gate_uses_date_distance_and_magnitude(self):
        event_date = MOD.date(2020, 1, 2)
        payload = {"features": [
            {"id": "ok", "properties": {"time": 1577923200000, "mag": 6.1, "place": "near"}, "geometry": {"coordinates": [10.1, 20.1, 12]}},
            {"id": "weak", "properties": {"time": 1577923200000, "mag": 3.1, "place": "near"}, "geometry": {"coordinates": [10.1, 20.1, 5]}},
        ]}
        rows = MOD.candidate_rows(payload, event_date, 10.0, 20.0)
        self.assertTrue(rows[0]["strict_candidate"])
        self.assertFalse(rows[1]["strict_candidate"])

    def test_multiple_candidates_never_auto_accept(self):
        event = {
            "source_collection": "PILD", "source_event_id": "e1", "event_date": "2020-01-02",
            "date_reliable": 1, "center_lon": 10.0, "center_lat": 20.0, "declared_usgs_event_id": "",
        }
        payload = {"features": [
            {"id": "a", "properties": {"time": 1577923200000, "mag": 6.0, "place": "a"}, "geometry": {"coordinates": [10.1, 20.1, 10]}},
            {"id": "b", "properties": {"time": 1577923200000, "mag": 5.8, "place": "b"}, "geometry": {"coordinates": [10.2, 20.2, 11]}},
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            def opener(*_, **__):
                return BytesResponse(json.dumps(payload).encode())
            result = MOD.audit_earthquake(event, Path(tmp), offline=False, refresh=False, opener=opener)
        self.assertEqual(result["support_status"], "needs_review")
        self.assertEqual(result["n_strict_candidates"], 2)

    def test_preregistered_id_still_requires_complete_shakemap(self):
        event = {
            "source_collection": "Sen12Landslides", "source_event_id": "e2", "event_date": "2020-01-02",
            "date_reliable": 1, "center_lon": 10.0, "center_lat": 20.0, "declared_usgs_event_id": "b",
        }
        query = {"features": [
            {"id": "a", "properties": {"time": 1577923200000, "mag": 6.0, "place": "a"}, "geometry": {"coordinates": [10.1, 20.1, 10]}},
            {"id": "b", "properties": {"time": 1577923200000, "mag": 6.2, "place": "b"}, "geometry": {"coordinates": [10.2, 20.2, 11]}},
        ]}
        detail = {"properties": {"products": {"shakemap": [{
            "source": "us", "status": "UPDATE", "preferredWeight": 10,
            "properties": {"review-status": "reviewed"},
            "contents": {"download/grid.xml": {"url": "https://example/grid.xml"}},
        }]}}}
        calls = iter([json.dumps(query).encode(), json.dumps(detail).encode(), b'<grid_field index="1" name="PGA"/><grid_field index="2" name="PGV"/><grid_field index="3" name="MMI"/>'])
        with tempfile.TemporaryDirectory() as tmp:
            def opener(*_, **__):
                return BytesResponse(next(calls))
            result = MOD.audit_earthquake(event, Path(tmp), offline=False, refresh=False, opener=opener)
        self.assertEqual(result["support_status"], "supported")
        self.assertEqual(result["match_basis"], "preregistered_usgs_event_id_verified")

    def test_missing_date_is_review_not_fabricated_support(self):
        event = {"event_date": None, "date_reliable": 0, "center_lon": 10, "center_lat": 20}
        with tempfile.TemporaryDirectory() as tmp:
            result = MOD.audit_earthquake(event, Path(tmp), offline=True, refresh=False)
        self.assertEqual(result["support_status"], "needs_review")

    def test_shakemap_product_prefers_reviewed_grid(self):
        detail = {"properties": {"products": {"shakemap": [
            {"preferredWeight": 99, "properties": {}, "contents": {"download/grid.xml": {}}},
            {"preferredWeight": 1, "properties": {"review-status": "reviewed"}, "contents": {"download/grid.xml": {}}},
        ]}}}
        self.assertEqual(MOD.choose_shakemap_product(detail)["preferredWeight"], 1)

    def test_grid_fields_are_case_normalized(self):
        payload = b'<grid_field index="1" name="PGA"/><grid_field index="2" name="PGV"/><grid_field index="3" name="MMI"/>'
        with tempfile.TemporaryDirectory() as tmp:
            fields = MOD.read_grid_header_fields(
                "https://example/grid", Path(tmp) / "header.xml", offline=False, refresh=False,
                opener=lambda *_, **__: BytesResponse(payload),
            )
        self.assertEqual(fields, {"pga", "pgv", "mmi"})

    @staticmethod
    def canonical_row(source, status, *, event_id="us-test", event_date="2020-01-02"):
        return {
            "source_collection": source,
            "source_event_id": source.lower(),
            "canonical_event_id": "canonical-1",
            "mechanism_family": "earthquake",
            "event_date": event_date,
            "date_reliable": 1,
            "center_lon": 10.0 if source == "PILD" else 10.2,
            "center_lat": 20.0 if source == "PILD" else 20.2,
            "support_status": status,
            "support_reason": "original",
            "usgs_event_id": event_id if status == "supported" else None,
            "earthquake_time_utc": "2020-01-02T01:00:00+00:00" if status == "supported" else None,
            "earthquake_event_lon": 10.1 if status == "supported" else None,
            "earthquake_event_lat": 20.1 if status == "supported" else None,
            "earthquake_magnitude": 6.5 if status == "supported" else None,
            "earthquake_depth_km": 12.0 if status == "supported" else None,
            "shakemap_has_pga": 1 if status == "supported" else None,
            "shakemap_has_pgv": 1 if status == "supported" else None,
            "shakemap_has_mmi": 1 if status == "supported" else None,
        }

    def test_canonical_support_propagates_to_nonconflicting_alias(self):
        frame = MOD.pd.DataFrame([
            self.canonical_row("PILD", "needs_review"),
            self.canonical_row("Sen12", "supported"),
        ])
        result = MOD.propagate_canonical_earthquake_support(frame)
        self.assertTrue(result["support_status"].eq("supported").all())
        propagated = result[result["source_collection"] == "PILD"].iloc[0]
        self.assertEqual(propagated["usgs_event_id"], "us-test")
        self.assertEqual(propagated["canonical_support_action"], "propagated_from_verified_alias")
        self.assertGreater(propagated["earthquake_distance_km"], 0)

    def test_canonical_official_id_conflict_blocks_entire_group(self):
        frame = MOD.pd.DataFrame([
            self.canonical_row("PILD", "supported", event_id="us-a"),
            self.canonical_row("Sen12", "supported", event_id="us-b"),
        ])
        result = MOD.propagate_canonical_earthquake_support(frame)
        self.assertTrue(result["support_status"].eq("needs_review").all())
        self.assertTrue(result["canonical_support_action"].eq("conflict_blocked").all())
        self.assertTrue(result["support_reason"].str.contains("usgs_event_id_conflict").all())

    def test_canonical_date_conflict_blocks_propagation(self):
        frame = MOD.pd.DataFrame([
            self.canonical_row("PILD", "needs_review", event_date="2020-01-10"),
            self.canonical_row("Sen12", "supported"),
        ])
        result = MOD.propagate_canonical_earthquake_support(frame)
        self.assertTrue(result["support_status"].eq("needs_review").all())
        self.assertTrue(result["support_reason"].str.contains("source_date_conflict").all())


if __name__ == "__main__":
    unittest.main()
