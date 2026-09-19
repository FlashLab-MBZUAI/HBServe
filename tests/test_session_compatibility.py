"""Current HBFSim receipt contracts; optional bounded native integration."""
from __future__ import annotations

import argparse
from copy import deepcopy
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hbfsim_client.simulation_session import (
    ResolvedSystemConfig, SimulationSession, SimulationSessionError,
    _validate_device_accounting,
)
from hbfsim_client.transaction_protocol import Transaction

SIMULATOR = None


def hbm_only_accounting():
    return {"hbm": {}, "hbf": None, "external": None, "host_dram": None,
            "base_die_link": {}, "hbf_external_direct_link": {}}


def wear_receipt():
    accounting = hbm_only_accounting()
    accounting["hbf"] = {
        "logical_write_bytes": 64, "physical_write_bytes": 0,
        "raw_physical_program_payload_bytes": 0, "data_program_payload_bytes": 0,
        "mapping_program_payload_bytes": 0, "gc_relocation_payload_bytes": 0,
        "waf_definition": "physical_write_bytes/(logical_write_bytes+raw_physical_program_payload_bytes)",
        "waf": 0.0,
        "state": {"writable_blocks": 2, "block_erase_count_sum": 3,
                  "accounting_verified": True},
    }
    return {
        "schema": {"name": "hbfsim.hbf_wear_snapshot", "version": 2},
        "result": "pass", "snapshot_id": "pending", "completed_frontier_ns": 10.0,
        "writable_blocks": 2, "block_erase_counts": [1, 2], "block_erase_count_sum": 3,
        "device_workload_totals": accounting,
        "quiescence": {"verified": False, "dirty_mapping_pages": 0,
                       "pending_dirty_mapping_events": 0, "pending_lpn_updates": 0,
                       "pending_vpn_updates": 0, "pending_commits": 0,
                       "write_buffer_entries": 1, "inflight_buffered_generations": 0,
                       "pending_physical_programs": 0, "pending_block_transitions": 0},
    }


def snapshot_from_receipt(receipt):
    # Exercise the public snapshot validation without an engine process.
    session = object.__new__(SimulationSession)
    session._closed = False
    session._enable_hbm = session._enable_hbf = True
    session._enable_external = session._enable_host_dram = False
    session._external_backing = None
    session._system_config = SimpleNamespace(
        hbf_geometry=SimpleNamespace(planes=1, blocks_per_plane=2))
    session._static_hbf_blocks_per_plane = 0
    session._wear_snapshot_ids = set()
    session._last_finish_ns = 10.0
    stream = io.StringIO()
    session._process = SimpleNamespace(stdin=stream)
    session._read_response = lambda description: deepcopy(receipt)
    try:
        result = session.hbf_wear_snapshot("pending")
        return result, stream.getvalue(), session.completed_frontier_ns
    finally:
        session._closed = True


class ReceiptContractTests(unittest.TestCase):
    def validate_hbm(self, value):
        _validate_device_accounting(value, enable_hbm=True, enable_hbf=False,
                                    enable_external=False, external_kind=None,
                                    description="HBM-only completion")

    def test_disabled_host_dram_has_explicit_null_accounting(self):
        self.validate_hbm(hbm_only_accounting())

    def test_missing_extra_and_unexpected_active_tiers_are_rejected(self):
        for change in ("missing", "extra", "active"):
            with self.subTest(change=change):
                value = hbm_only_accounting()
                if change == "missing":
                    del value["host_dram"]
                elif change == "extra":
                    value["unrecognized_tier"] = None
                else:
                    value["host_dram"] = {}
                with self.assertRaises(SimulationSessionError):
                    self.validate_hbm(value)

    def test_wear_v2_observes_pending_work_without_advancing_time(self):
        result, command, frontier = snapshot_from_receipt(wear_receipt())
        self.assertEqual(command, "WEAR_SNAPSHOT pending\n")
        self.assertEqual(frontier, 10.0)
        self.assertFalse(result["quiescence"]["verified"])
        self.assertEqual(result["quiescence"]["write_buffer_entries"], 1)

    def test_wear_rejects_inconsistent_physical_and_pending_state(self):
        for change in ("sum", "blocks", "verified", "pending", "old_schema", "missing_tier"):
            with self.subTest(change=change):
                value = wear_receipt()
                state = value["device_workload_totals"]["hbf"]["state"]
                if change == "sum":
                    state["block_erase_count_sum"] += 1
                elif change == "blocks":
                    state["writable_blocks"] += 1
                elif change == "verified":
                    state["accounting_verified"] = False
                elif change == "pending":
                    value["quiescence"]["verified"] = True
                elif change == "old_schema":
                    value["schema"]["version"] = 1
                else:
                    del value["device_workload_totals"]["host_dram"]
                with self.assertRaises(SimulationSessionError):
                    snapshot_from_receipt(value)


class NativeSessionTests(unittest.TestCase):
    def setUp(self):
        if SIMULATOR is None:
            self.skipTest("pass --simulator for native session checks")

    def config(self, *overlays):
        return ResolvedSystemConfig.load((
            ROOT / "configs/systems/eight-stack-baseline.cfg",
            ROOT / "configs/systems/sglang-small.cfg", *overlays))

    def session(self, *, config=None, **options):
        return SimulationSession(simulator_path=SIMULATOR,
                                 system_config=config or self.config(),
                                 read_timeout_s=10, **options)

    def test_hbm_only_completion_and_stop_include_disabled_host_dram(self):
        with self.session(enable_hbm=True, enable_hbf=False) as session:
            result = session.run((Transaction("read", "HBM", "R", 0, 64, 0.0),))
            self.assertIsNone(result.receipt["device_delta"]["host_dram"])
            self.assertEqual(result.receipt["by_target"]["HOST_DRAM"],
                             {"transactions": 0, "bytes": 0})
        self.assertIsNone(session.source_receipt()["execution_options"]["host_dram_backing"])

    def test_host_dram_and_nvme_preserve_separate_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            overlay = Path(directory) / "host-dram.cfg"
            # Reuse fixture timing; this checks accounting, not DRAM latency.
            overlay.write_text((ROOT / "configs/nvme-ssd.cfg").read_text().replace(
                "external-backing-kind=nvme-ssd", "external-backing-kind=host-dram"))
            host_config = self.config(overlay)
            with self.session(config=self.config(ROOT / "configs/nvme-ssd.cfg"),
                              enable_hbm=True, enable_hbf=False, enable_external=True,
                              host_dram_config=host_config) as session:
                result = session.run((
                    Transaction("host", "HOST_DRAM", "R", 4032, 128, 0.0),
                    Transaction("ssd", "EXTERNAL", "W", 0, 4096, 0.0),
                ))
                host = result.receipt["device_delta"]["host_dram"]
                ssd = result.receipt["device_delta"]["external"]
                self.assertEqual((host["read_bytes"], host["write_bytes"]), (128, 0))
                self.assertEqual((ssd["read_bytes"], ssd["write_bytes"]), (0, 4096))
                self.assertEqual(host["page_run_pages"], 2)
                self.assertEqual(host["page_run_segments"], 2)
                corrupted = deepcopy(result.receipt["device_delta"])
                corrupted["host_dram"]["read_bytes"] += 1
                with self.assertRaises(SimulationSessionError):
                    _validate_device_accounting(
                        corrupted, enable_hbm=True, enable_hbf=False,
                        enable_external=True, enable_host_dram=True,
                        external_kind="nvme-ssd", description="corrupted host traffic")
            self.assertEqual(session.source_receipt()["execution_options"]["host_dram_backing"]["kind"], "host-dram")

    def test_native_wear_observation_and_checkpoint(self):
        with self.session(enable_hbm=True, enable_hbf=True) as session:
            session.run((Transaction("write", "HBF_LOGICAL", "W", 0, 64, 0.0),))
            frontier = session.completed_frontier_ns
            pending = session.hbf_wear_snapshot("pending")
            self.assertEqual(pending["schema"]["version"], 2)
            self.assertEqual(session.completed_frontier_ns, frontier)
            self.assertGreater(pending["quiescence"]["write_buffer_entries"], 0)
            session.checkpoint("flush")
            flushed = session.hbf_wear_snapshot("flushed")
            self.assertTrue(flushed["quiescence"]["verified"])
            self.assertEqual(flushed["block_erase_count_sum"], sum(flushed["block_erase_counts"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", type=Path)
    args, rest = parser.parse_known_args()
    if args.simulator is not None:
        SIMULATOR = args.simulator.resolve()
        if not SIMULATOR.is_file():
            parser.error(f"simulator does not exist: {SIMULATOR}")
    unittest.main(argv=[__file__, *rest])
