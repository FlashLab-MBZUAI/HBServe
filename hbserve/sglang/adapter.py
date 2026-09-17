"""Hooks for the pinned upstream simulator; imported in spawned workers too."""
from __future__ import annotations

import atexit
from dataclasses import asdict
import json
import os
from pathlib import Path

from .backend import HardwareSession
from .model import DTYPE_NAMES, NativeRequest

_released_slots = []


def capture_batch(batch, future_map):
    seq_lens = batch.seq_lens_cpu.tolist()
    pool_indices = batch.req_pool_indices_cpu.tolist()
    if not (len(seq_lens) == len(pool_indices) == len(batch.reqs)):
        raise ValueError("native sequence/slot map does not match the request batch")
    mixed = getattr(batch, "mix_running_indices_cpu", None)
    mixed_pools = set(mixed.tolist()) if mixed is not None else set()
    if batch.forward_mode.is_extend():
        token_ids = batch.prefill_input_ids_cpu.tolist()
        mixed_device = getattr(batch, "mix_running_indices", None)
        if mixed_device is not None:
            token_ids += future_map.output_tokens_buf[mixed_device].tolist()
    elif batch.forward_mode.is_decode():
        token_ids = future_map.output_tokens_buf[batch.req_pool_indices].tolist()
    else:
        raise ValueError("native HBFSim adapter supports extend and decode only")
    offset, rows = 0, []
    for i, req in enumerate(batch.reqs):
        decode = batch.forward_mode.is_decode() or pool_indices[i] in mixed_pools
        extend = 1 if decode else int(batch.extend_lens[i])
        slots = batch.req_to_token_pool.req_to_token[pool_indices[i], :seq_lens[i]].tolist()
        row = NativeRequest(str(req.rid), tuple(map(int, slots)),
            tuple(map(int, token_ids[offset:offset+extend])), len(req.origin_input_ids),
            int(req.sampling_params.max_new_tokens), "decode" if decode else "prefill",
            bool(decode or seq_lens[i] >= len(req.full_untruncated_fill_ids)))
        rows.append(asdict(row))
        offset += extend
    if offset != len(token_ids):
        raise ValueError("native input tokens disagree with extend lengths")
    return rows


def install_allocator_observer():
    from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
    if getattr(TokenToKVPoolAllocator, "_hbfsim_observed", False):
        return
    token_free = TokenToKVPoolAllocator.free
    page_release = PagedTokenToKVPoolAllocator._release_page_ids

    def free(self, indices):
        actual = self.free_group is None
        result = token_free(self, indices)
        if actual:
            _released_slots.extend(map(int, indices.tolist()))
        return result

    def release(self, *page_ids):
        result = page_release(self, *page_ids)
        for ids in page_ids:
            for page in ids.tolist():
                _released_slots.extend(range(int(page)*self.page_size, (int(page)+1)*self.page_size))
        return result

    TokenToKVPoolAllocator.free = free
    PagedTokenToKVPoolAllocator._release_page_ids = release
    TokenToKVPoolAllocator._hbfsim_observed = True


def take_released_slots():
    result = sorted(set(_released_slots))
    _released_slots.clear()
    return result


class HBFSimPredictor:
    def __init__(self, model, hw, scheduler, config, output):
        if any(getattr(scheduler, k, 1) != 1 for k in ("tp_size", "pp_size", "dp_size", "ep_size", "cp_size")):
            raise ValueError("HBFSim native serving models one worker; collectives need a separate graph")
        if getattr(scheduler, "enable_hierarchical_cache", False):
            raise ValueError("SGLang HiCache is separate from the modeled HBM/HBF cache and is not enabled")
        if getattr(model, "attention_arch", "MHA") not in (None, "MHA"):
            raise ValueError("the native traffic compiler requires dense MHA/GQA")
        self.hardware = HardwareSession(config, output)
        try:
            hf = config["model"]
            expected = (self.hardware.model.num_layers, hf["num_attention_heads"],
                hf.get("num_key_value_heads", hf["num_attention_heads"]),
                hf.get("head_dim", hf["hidden_size"] // hf["num_attention_heads"]))
            actual = (model.num_hidden_layers, model.num_attention_heads,
                      model.num_key_value_heads, model.head_dim)
            if actual != expected or model.hidden_size != hf["hidden_size"]:
                raise ValueError("native model geometry differs from the HBFSim traffic model")
            if scheduler.data_type.name != DTYPE_NAMES[config["dtype"]]:
                raise ValueError("native weight dtype differs from the traffic model")
            kv_dtype = scheduler.kv_cache_data_type or scheduler.data_type
            kv_bytes = model.num_key_value_heads * (model.head_dim + (model.v_head_dim or model.head_dim)) * kv_dtype.bytes
            if kv_bytes != self.hardware.kv_bytes:
                raise ValueError("native KV dtype/geometry differs from the physical byte ledger")
            if scheduler.max_total_tokens != self.hardware.max_total_tokens:
                raise ValueError("native admission limit differs from the physical capacity budget")
        except BaseException:
            self.hardware.close()
            raise
        atexit.register(self.hardware.close)

    def predict_infer_time(self, batch):
        from sglang_simulator.simulation.manager import StateManager
        snapshot = getattr(batch, "hbfsim_requests", None)
        if snapshot is None or len(snapshot) != len(batch.reqs):
            raise RuntimeError("missing native slot snapshot; prepare the pinned SGLang checkout")
        rows = [NativeRequest(**{**r, "slots": tuple(r["slots"]), "tokens": tuple(r["tokens"])}) for r in snapshot]
        start_ns = StateManager.get_global_clock() * 1e9
        result = self.hardware.run_batch(rows, now_ns=start_ns, freed_slots=take_released_slots())
        return result["latency_ns"] / 1e9

    def finalize(self):
        if not self.hardware.closed:
            from sglang_simulator.simulation.manager import StateManager
            now_ns = StateManager.get_global_clock() * 1e9
            self.hardware._align_clock(now_ns)
            self.hardware.last_serving_finish_ns = now_ns
            self.hardware.finalize(take_released_slots())

    def get_metrics(self):
        return {"hbfsim_batches": self.hardware.batches,
            "hbfsim_result": str(self.hardware.output / "result.json"),
            "hbfsim_calibrated": False}

    def reset_metrics(self):
        # Upstream profile flushes must never reset device or allocation state.
        pass


def install():
    from sglang_simulator.simulation.manager.config import ConfigManager
    if getattr(ConfigManager, "_hbfsim_installed", False):
        return
    original = ConfigManager.get_inference_time_predictor

    def factory(cls, model, hw, scheduler):
        settings = cls._get_raw_config().get("predictor", {})
        if settings.get("name") != "hbfsim":
            return original(model, hw, scheduler)
        path = Path(cls.resolve_config_relative_path(settings["hardware_config"]))
        install_allocator_observer()
        return HBFSimPredictor(model, hw, scheduler, json.loads(path.read_text()),
            Path(os.environ["SGLANG_SIMULATOR_OUTPUT_DIR"]) / "hbfsim")

    ConfigManager.get_inference_time_predictor = classmethod(factory)
    ConfigManager._hbfsim_installed = True
