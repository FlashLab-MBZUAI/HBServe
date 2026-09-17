"""Trace identity, prefix semantics and native completion reconciliation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from hbserve.sglang.requests import (
    bind_request_ids, encode_trace, load_requests, request_stats_on_input_clock, validate_request_stats,
)
from workloads.production_request_trace.canonical import ProductionTrace, RECORD_SCHEMA, publish_trace
from workloads.production_request_trace.registry import load_source_spec


def fixture(block=16, *, synthetic=False):
    source_id = "qwen_bailian_trace_a" if block == 16 else (
        "mooncake_fast25_synthetic" if synthetic else "mooncake_fast25_conversation")
    source = load_source_spec(source_id).manifest_source()
    rows = []
    for i, (hashes, length) in enumerate((([7, 8], 2*block), ([7, 9], block+3),
            ([7, 8], 2*block), ([10, 8], 2*block), ([7], block-1), ([7, 8, 8], 3*block))):
        rows.append({"schema": RECORD_SCHEMA, "request_id": f"{source_id}:{i}", "source_index": i,
            "arrival_ns": (i//2)*1000000, "input_tokens": length, "output_tokens": i+1,
            "prefix_hash_ids": hashes, "session_id": f"{source_id}:{i}" if block == 16 else None,
            "parent_request_id": None, "turn": 1 if block == 16 else None,
            "request_type": "text" if block == 16 else None})
    return ProductionTrace(Path("unused"), {"source": source}, tuple(rows), "a"*64, "b"*64)


def common_tokens(left, right):
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))


class TraceInputTests(unittest.TestCase):
    def test_exact_block_prefixes_for_both_sources_and_partial_tails(self):
        for block in (16, 512):
            with self.subTest(block=block):
                trace = fixture(block)
                result = encode_trace(trace, 256)
                for original, row in zip(trace.records, result.records):
                    self.assertEqual(row["request_id"], original["request_id"])
                    self.assertEqual(row["arrival_ns"], original["arrival_ns"])
                    self.assertEqual(row["output_tokens"], original["output_tokens"])
                    self.assertEqual(len(row["token_ids"]), original["input_tokens"])
                    self.assertTrue(all(2 <= t < 256 for t in row["token_ids"]))
                for i, left in enumerate(trace.records):
                    for j, right in enumerate(trace.records):
                        shared_blocks = common_tokens(left["prefix_hash_ids"], right["prefix_hash_ids"])
                        expected = min(shared_blocks*block, left["input_tokens"], right["input_tokens"])
                        self.assertEqual(common_tokens(result.records[i]["token_ids"],
                                                       result.records[j]["token_ids"]), expected)
                self.assertEqual(result.records, encode_trace(trace, 256).records)

    def test_selection_rebases_once_preserves_ties_ids_and_parent_metadata(self):
        trace = fixture()
        trace.records[3]["parent_request_id"] = trace.records[0]["request_id"]
        result = encode_trace(trace, 256, start=2, count=3)
        self.assertEqual([r["arrival_ns"] for r in result.records], [0, 0, 1000000])
        self.assertEqual([r["source_index"] for r in result.records], [2, 3, 4])
        self.assertEqual(result.source_records[1]["parent_request_id"], trace.records[0]["request_id"])
        self.assertEqual(result.provenance["selection"]["arrival_origin_ns"], 1000000)
        for options in ({"start": 6}, {"start": -1}, {"count": 0}):
            with self.assertRaises(ValueError):
                encode_trace(trace, 256, **options)

    def test_no_vocabulary_modulo_alias_and_no_dummy_output_prefix(self):
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            encode_trace(fixture(), 3)
        result = encode_trace(fixture(), 4)
        self.assertEqual(common_tokens(result.records[0]["token_ids"] + [1]*16,
                                       result.records[5]["token_ids"]), 32)

    def test_unknown_partial_block_overlap_is_not_invented(self):
        trace = fixture()
        rows = deepcopy(trace.records[:2])
        for row, hash_id in zip(rows, (100, 101)):
            row.update(input_tokens=3, prefix_hash_ids=[hash_id])
        result = encode_trace(replace(trace, records=rows), 256)
        self.assertEqual(common_tokens(*(r["token_ids"] for r in result.records)), 0)

    def test_synthetic_source_remains_explicit(self):
        with self.assertRaisesRegex(ValueError, "allow-synthetic"):
            encode_trace(fixture(512, synthetic=True), 256)
        result = encode_trace(fixture(512, synthetic=True), 256, allow_synthetic=True)
        self.assertFalse(result.provenance["source"]["production_eligible"])

    def test_canonical_bundle_loading_and_frozen_evidence(self):
        trace = fixture()
        transform = {"operation": "source_import", "parent_manifest_sha256": None,
            "parent_requests_sha256": None, "slice_start": 0, "slice_stop": len(trace.records),
            "time_scale_numerator": 1, "time_scale_denominator": 1, "selected_origin_ns": 0,
            "rebase_to_zero": True,
            "timestamp_formula": "floor((selected_arrival_ns-selected_origin_ns)*time_scale_numerator/time_scale_denominator)"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            canonical = publish_trace(path / "bundle", trace_id="test-fixture", source=trace.manifest["source"],
                transform=transform, records=trace.records)
            result = load_requests(canonical.directory, 256, start=2, count=2)
            result.freeze(path)
            reloaded = load_requests(path / "requests.jsonl", 256)
            self.assertEqual(result.records, reloaded.records)
            evidence = json.loads((path / "request-provenance.json").read_text())
            self.assertEqual(evidence["source_requests_sha256"], canonical.requests_sha256)
            self.assertEqual(hashlib.sha256((path / "trace-manifest.json").read_bytes()).hexdigest(),
                             canonical.manifest_sha256)
            self.assertEqual(len((path / "trace-records.jsonl").read_text().splitlines()), 2)
            self.assertTrue((path / "trace-manifest.json").is_file())

    def test_native_input_keeps_nonzero_arrival_and_rejects_raw_without_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.jsonl"
            row = {"arrival_ns": 20000, "token_ids": [1, 2], "output_tokens": 3}
            path.write_text(json.dumps(row)+"\n")
            self.assertEqual(load_requests(path, 4).records[0]["arrival_ns"], 20000)
            with self.assertRaisesRegex(ValueError, "selection options"):
                load_requests(path, 4, count=1)
            path.write_text(json.dumps({"hash_ids": [0]})+"\n")
            with self.assertRaisesRegex(ValueError, "trace-source-id"):
                load_requests(path, 4)

    def test_dispatch_and_completion_keep_source_identity_and_exact_lengths(self):
        rows = encode_trace(fixture(), 256, count=2).records
        engine = SimpleNamespace(async_generate=lambda **kwargs: kwargs)
        bind_request_ids(engine, rows)
        self.assertEqual([engine.async_generate()["rid"] for _ in rows], [r["request_id"] for r in rows])
        stats = [{"rid": r["request_id"], "created_time": r["arrival_ns"]/1e9,
            "input_length": len(r["token_ids"]), "output_length": r["output_tokens"],
            "gen_token_latencies": [0.001]*r["output_tokens"], "queue_start": 0,
            "queue_end": 0, "last_event_time": 1} for r in rows]
        validate_request_stats(rows, stats[::-1])
        delayed = [{**r, "arrival_ns": r["arrival_ns"]+20000} for r in rows]
        restored = request_stats_on_input_clock(stats, 20000)
        validate_request_stats(delayed, restored)
        self.assertEqual(restored[0]["created_time"], 0.00002)
        self.assertEqual(stats[0]["created_time"], 0)
        for field, value in (("rid", "wrong"), ("created_time", 1), ("input_length", 1),
                             ("output_length", 100), ("gen_token_latencies", [])):
            invalid = deepcopy(stats)
            invalid[0][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                validate_request_stats(rows, invalid)


if __name__ == "__main__":
    unittest.main()
