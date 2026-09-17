"""Physical integration checks for native-slot liveness, capacity and timing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

from hbserve.sglang.backend import HardwareSession, interval_bytes
from hbserve.sglang.model import NativeCompiler, NativeRequest, dense_model
from hbserve.contracts import RooflineTimingProvider

ROOT = Path(__file__).resolve().parents[1]
HBFSIM = ROOT.parent / "HBFSim"
SIMULATOR = HBFSIM / "build/hbfsim"


def request(rid, slots, tokens, *, prompt=None, output=8, phase="prefill", emits=True):
    return NativeRequest(rid, tuple(slots), tuple(tokens), prompt or len(slots), output, phase, emits)


class NativeBackendTest(unittest.TestCase):
    def setUp(self):
        if self._testMethodName != "test_actual_token_ids_and_chunk_output_semantics" and not SIMULATOR.is_file():
            self.skipTest("optional physical integration requires a built HBFSim executable")

    def config(self, **options):
        return {"model": json.loads((ROOT / "examples/sglang/tiny-qwen3/config.json").read_text()),
            "dtype": "bfloat16", "kv_dtype": "bfloat16", "simulator": str(SIMULATOR),
            "system_configs": [str(HBFSIM / "configs/systems/eight-stack-baseline.cfg"),
                               str(HBFSIM / "configs/systems/sglang-small.cfg")],
            "compute": {"peak_tflops": 100, "efficiency": 0.5},
            "weight_tier": "hbf", "kv_tier": "hbf", "architecture": "tiered",
            "max_total_tokens": 128, "page_size": 4, "kv_cache_bytes": 8192,
            "background_writeback_pages": 0, **options}

    def session(self, **options):
        self.assertTrue(SIMULATOR.is_file(), f"build HBFSim or pass --simulator: {SIMULATOR}")
        session = HardwareSession(self.config(**options))
        self.addCleanup(session.close)
        return session

    def test_shared_prefix_survives_other_request_free(self):
        s = self.session()
        first = s.run_batch([request("a", (4, 5), (10, 11))], now_ns=0)
        s.run_batch([request("b", (4, 5, 6), (12,))], now_ns=first["forward_finish_ns"])
        s.release((6, 7))  # One written token and never-written page padding.
        self.assertEqual(s.live_slots, {4, 5})
        self.assertEqual(s.counters["hbf_invalidated_pages"], 0)
        self.assertEqual(sum(interval_bytes(e.dirty) for e in s.cache.values()), 512)
        self.assertEqual(s.counters["discarded_dirty_bytes"], 256)
        s.finalize()
        self.assertEqual(s.counters["writeback_bytes"], 512)
        self.assertEqual(s.report()["dirty_cache_bytes"], 0)

    def test_dead_dirty_tokens_are_discarded_without_destage(self):
        s = self.session()
        s.run_batch([request("a", range(4, 8), range(10, 14))], now_ns=0)
        s.finalize(range(4, 8))
        self.assertEqual(s.counters["writeback_bytes"], 0)
        self.assertEqual(s.counters["discarded_dirty_bytes"], 1024)
        self.assertEqual(len(s.cache), 0)

    def test_page_eviction_writes_back_and_reads_real_backing(self):
        s = self.session(kv_cache_bytes=4096)
        first = s.run_batch([request("a", range(4, 16), range(10, 22), prompt=12)], now_ns=0)
        s.run_batch([request("a", range(4, 17), (30,), prompt=12, phase="decode")],
                    now_ns=first["forward_finish_ns"])
        self.assertGreater(s.counters["cache_evictions"], 0)
        self.assertEqual(s.counters["cache_miss_bytes"], 12*128*2)
        s.finalize(range(4, 17))
        self.assertGreater(s.counters["writeback_bytes"], 0)
        self.assertEqual(s.counters["hbf_invalidated_pages"], 2)

    def test_partial_page_free_does_not_invalidate_live_backing(self):
        s = self.session(kv_cache_bytes=0, architecture="peer")
        first = s.run_batch([request("a", (4, 5), (10, 11))], now_ns=0)
        s.release((4,))
        self.assertEqual(s.counters["hbf_invalidated_pages"], 0)
        s.run_batch([request("b", (5, 6), (12,))], now_ns=first["forward_finish_ns"])
        s.finalize((5, 6))
        self.assertEqual(s.counters["hbf_invalidated_pages"], 2)

    def test_nonzero_arrival_is_excluded_from_forward_latency(self):
        s = self.session()
        r = s.run_batch([request("a", (4, 5), (10, 11))], now_ns=1e8)
        self.assertGreater(r["forward_finish_ns"], 1e8)
        self.assertAlmostEqual(r["latency_ns"], r["forward_finish_ns"] - 1e8)
        self.assertLess(r["latency_ns"], 1e7)
        with self.assertRaisesRegex(ValueError, "backwards"):
            s.run_batch([request("b", (6,), (12,))], now_ns=0)

    def test_background_writeback_is_detached_but_reuse_waits(self):
        with tempfile.TemporaryDirectory() as temp:
            overlay = Path(temp) / "flush.cfg"
            overlay.write_text("hbf-write-buffer-completion-requires-flush=true\nhbf-program-ns=1000000\n")
            # HBM weights isolate detached work; HBF weights can legitimately
            # wait on the same NAND resources as the background programs.
            config = self.config(background_writeback_pages=2, weight_tier="hbm")
            config["system_configs"].append(str(overlay))
            s = HardwareSession(config)
            try:
                first = s.run_batch([request("a", (4, 5), (10, 11))], now_ns=0)
                self.assertGreater(first["issued_finish_ns"], first["forward_finish_ns"])
                second = s.run_batch([request("a", (4, 5, 6), (12,), prompt=2, phase="decode")],
                    now_ns=first["forward_finish_ns"])
                self.assertGreaterEqual(second["forward_finish_ns"], first["issued_finish_ns"])
                s.finalize((4, 5, 6))
            finally:
                s.close()

    def test_capacity_includes_reserved_slots_and_controller(self):
        config = self.config()
        layout = HardwareSession(config, initialize=False)
        self.assertEqual(layout.pool_tokens, 132)
        self.assertGreater(layout.budget["controller_reserve_bytes"], 0)
        for tier, used in layout.budget["allocated_bytes"].items():
            self.assertLessEqual(used, layout.budget["capacity_bytes"][tier])
        with self.assertRaises(ValueError):
            HardwareSession(self.config(max_total_tokens=10**9), initialize=False)

    def test_hbm_resident_and_fp8_geometry(self):
        s = self.session(weight_tier="hbm", kv_tier="hbm", kv_cache_bytes=0, kv_dtype="fp8_e4m3")
        r = s.run_batch([request("a", (4, 5), (10, 11))], now_ns=0)
        self.assertEqual(r["traffic"]["kv_write_bytes"], 2*64*2)
        self.assertFalse(s.enable_hbf)
        s.finalize((4, 5))

    def test_actual_token_ids_and_chunk_output_semantics(self):
        hf = self.config()["model"]
        model = dense_model(hf, "bfloat16", "bfloat16")
        row = request("a", (4, 5), (37, 41), prompt=4, emits=False)
        compiled = NativeCompiler(model, [row], RooflineTimingProvider(100, 0.5)).batch(0, 0)
        embeddings = [op for op in compiled.operations if op.role == "embedding/read"]
        self.assertEqual([op.offset for op in embeddings], [37*256, 41*256])
        self.assertFalse(any(op.role == "lm_head/read" for op in compiled.operations))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--simulator", type=Path, default=SIMULATOR)
    p.add_argument("--hbfsim-root", type=Path, default=HBFSIM)
    args, rest = p.parse_known_args()
    SIMULATOR, HBFSIM = args.simulator.resolve(), args.hbfsim_root.resolve()
    if not SIMULATOR.is_file():
        p.error(f"build HBFSim or pass an existing --simulator: {SIMULATOR}")
    unittest.main(argv=[__file__, *rest])
