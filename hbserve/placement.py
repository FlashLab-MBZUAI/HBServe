#!/usr/bin/env python3
"""Capacity-checked placement of HBServe objects onto HBFSim tiers.

Weights are placed statically by tier (with per-object overrides) or cached
whole-model in HBM.  KV is paged: every request owns one block table per
layer, blocks are allocated incrementally as tokens are processed, and blocks
of requests that are not running may migrate whole-request to a cold tier
(HBF or external) and back on demand as explicit transactions.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from hbserve.prefix import PrefixCache, block_keys
from hbserve.block_references import BlockReferences
import heapq
import math
from typing import Any, Mapping, Sequence

from hbfsim_client.transaction_protocol import (
    HbfGeometry,
    Transaction,
    TransactionBatch,
    hbf_link_bytes_by_stack,
)
from hbserve.contracts import (
    BatchSlice,
    CanonicalServingBatch,
    MemoryObject,
    ModelSpec,
    RequestSpec,
    HBServeError,
    canonical_sha256,
)


PLACEMENT_SCHEMA = {
    "name": "hbserve.placement",
    "version": 2,
}
REMAP_SCHEMA = {
    "name": "hbserve.remap_receipt",
    "version": 2,
}
WEIGHT_TIERS = {
    "hbm",
    "hbf",
    "external_direct",
    "external_cached_hbm",
}
KV_HOT_TIER = "hbm"
KV_COLD_TIERS = {"hbf", "external"}
KV_ALLOCATION_POLICY = "paged_blocks_incremental_v1"
KV_MIGRATION_POLICY = "whole_request_coldest_first_v1"
TIER_TARGETS = {
    "hbm": "HBM",
    "hbf": "HBF_LOGICAL",
    "external": "EXTERNAL",
}


def _positive_integer(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HBServeError(f"{description} must be an integer > 0")
    return value


def _nonnegative_integer(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HBServeError(f"{description} must be an integer >= 0")
    return value


def _align_up(value: int, alignment: int) -> int:
    _nonnegative_integer(value, "alignment value")
    _positive_integer(alignment, "alignment")
    if alignment & (alignment - 1):
        raise HBServeError("alignment must be a power of two")
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class KvPlacement:
    """Object-class rule for KV: running requests are hot, others may go cold."""

    hot: str = KV_HOT_TIER
    cold: str | None = None

    def __post_init__(self) -> None:
        if self.hot != KV_HOT_TIER:
            raise HBServeError("hot KV must live in hbm")
        if self.cold is not None and self.cold not in KV_COLD_TIERS:
            raise HBServeError(
                f"cold KV tier must be one of {sorted(KV_COLD_TIERS)} or null"
            )

    def canonical(self) -> dict[str, Any]:
        return {"hot": self.hot, "cold": self.cold}


@dataclass(frozen=True)
class PlacementSpec:
    """One explicit capacity, weight-residency, and KV-placement policy."""

    hbm_capacity_bytes: int
    hbf_capacity_bytes: int
    external_capacity_bytes: int
    hbm_runtime_reserve_bytes: int
    hbm_model_cache_bytes: int
    model_weight_tiers: Mapping[str, str]
    object_tier_overrides: Mapping[str, str]
    model_load_chunk_bytes: int = 16 * 1024 * 1024
    initial_cached_models: tuple[str, ...] = ()
    hbm_alignment_bytes: int = 4096
    hbf_page_size_bytes: int = 4096
    external_page_size_bytes: int = 4096
    kv_block_tokens: int = 16
    kv_placement: KvPlacement = field(default_factory=KvPlacement)
    prefix_cache_bytes: int = 0
    prefix_cache_ttl_ns: float | None = None

    def __post_init__(self) -> None:
        _positive_integer(self.hbm_capacity_bytes, "HBM capacity")
        _nonnegative_integer(self.hbf_capacity_bytes, "HBF capacity")
        _nonnegative_integer(self.external_capacity_bytes, "external capacity")
        _nonnegative_integer(
            self.hbm_runtime_reserve_bytes, "HBM runtime reserve"
        )
        _nonnegative_integer(self.hbm_model_cache_bytes, "HBM model cache")
        _positive_integer(self.kv_block_tokens, "KV block tokens")
        _nonnegative_integer(self.prefix_cache_bytes, "prefix cache bytes")
        if self.prefix_cache_ttl_ns is not None and (
            isinstance(self.prefix_cache_ttl_ns, bool)
            or not isinstance(self.prefix_cache_ttl_ns, (int, float))
            or not math.isfinite(self.prefix_cache_ttl_ns) or self.prefix_cache_ttl_ns <= 0
        ):
            raise HBServeError("prefix cache TTL must be finite and positive")
        if self.prefix_cache_ttl_ns is not None and self.prefix_cache_bytes == 0:
            raise HBServeError("prefix cache TTL requires a nonzero cache budget")
        if not isinstance(self.kv_placement, KvPlacement):
            raise HBServeError("kv_placement must be a KvPlacement")
        for name, value in (
            ("HBM alignment", self.hbm_alignment_bytes),
            ("HBF page size", self.hbf_page_size_bytes),
            ("external page size", self.external_page_size_bytes),
            ("model load chunk", self.model_load_chunk_bytes),
        ):
            _positive_integer(value, name)
            if value & (value - 1):
                raise HBServeError(f"{name} must be a power of two")
        if (
            self.model_load_chunk_bytes % self.hbm_alignment_bytes
            or self.model_load_chunk_bytes % self.external_page_size_bytes
        ):
            raise HBServeError(
                "model load chunks must align to HBM and external pages"
            )
        for name, value in (
            ("HBM capacity", self.hbm_capacity_bytes),
            ("HBM runtime reserve", self.hbm_runtime_reserve_bytes),
            ("HBM model cache", self.hbm_model_cache_bytes),
        ):
            if value % self.hbm_alignment_bytes:
                raise HBServeError(
                    f"{name} must align to the HBM allocation granularity"
                )
        if self.hbf_capacity_bytes % self.hbf_page_size_bytes:
            raise HBServeError(
                "HBF capacity must align to the HBF page size"
            )
        if self.external_capacity_bytes % self.external_page_size_bytes:
            raise HBServeError(
                "external capacity must align to the external page size"
            )
        if not self.model_weight_tiers:
            raise HBServeError("model_weight_tiers must not be empty")
        for model_id, tier in self.model_weight_tiers.items():
            if not isinstance(model_id, str) or not model_id:
                raise HBServeError("placement model ID is empty")
            if tier not in WEIGHT_TIERS:
                raise HBServeError(
                    f"unsupported weight tier for {model_id}: {tier!r}"
                )
        for object_id, tier in self.object_tier_overrides.items():
            if not isinstance(object_id, str) or not object_id:
                raise HBServeError("placement object ID is empty")
            if tier not in WEIGHT_TIERS - {"external_cached_hbm"}:
                raise HBServeError(
                    "object overrides support hbm, hbf, or external_direct"
                )
        if len(self.initial_cached_models) != len(set(self.initial_cached_models)):
            raise HBServeError("initial_cached_models contains duplicates")
        if self.kv_placement.cold == "hbf" and self.hbf_capacity_bytes == 0:
            raise HBServeError("cold KV in HBF requires HBF capacity")
        if (
            self.kv_placement.cold == "external"
            and self.external_capacity_bytes == 0
        ):
            raise HBServeError(
                "cold KV in the external tier requires external capacity"
            )

    @property
    def kv_cold_page_bytes(self) -> int | None:
        if self.kv_placement.cold == "hbf":
            return self.hbf_page_size_bytes
        if self.kv_placement.cold == "external":
            return self.external_page_size_bytes
        return None

    def canonical(self) -> dict[str, Any]:
        return {
            "schema": PLACEMENT_SCHEMA,
            "hbm_capacity_bytes": self.hbm_capacity_bytes,
            "hbf_capacity_bytes": self.hbf_capacity_bytes,
            "external_capacity_bytes": self.external_capacity_bytes,
            "hbm_runtime_reserve_bytes": self.hbm_runtime_reserve_bytes,
            "hbm_model_cache_bytes": self.hbm_model_cache_bytes,
            "model_weight_tiers": dict(sorted(self.model_weight_tiers.items())),
            "object_tier_overrides": dict(
                sorted(self.object_tier_overrides.items())
            ),
            "model_load_chunk_bytes": self.model_load_chunk_bytes,
            "initial_cached_models": list(self.initial_cached_models),
            "hbm_alignment_bytes": self.hbm_alignment_bytes,
            "hbf_page_size_bytes": self.hbf_page_size_bytes,
            "external_page_size_bytes": self.external_page_size_bytes,
            "kv_block_tokens": self.kv_block_tokens,
            "kv_placement": self.kv_placement.canonical(),
            "prefix_cache_bytes": self.prefix_cache_bytes,
            "prefix_cache_ttl_ns": self.prefix_cache_ttl_ns,
        }

    @cached_property
    def digest(self) -> str:
        return canonical_sha256(self.canonical())


@dataclass(frozen=True)
class ObjectAddress:
    object_id: str
    target: str
    addr: int
    bytes: int

    def canonical(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "target": self.target,
            "addr": self.addr,
            "bytes": self.bytes,
        }


class _ExtentAllocator:
    """Exact first-fit allocator with coalescing and no hidden overcommit."""

    def __init__(self, begin: int, end: int, alignment: int) -> None:
        if begin < 0 or end < begin:
            raise HBServeError("extent allocator range is invalid")
        self.begin = begin
        self.end = end
        self.alignment = alignment
        self._free: list[tuple[int, int]] = (
            [(begin, end - begin)] if end > begin else []
        )

    @property
    def free_bytes(self) -> int:
        return sum(length for _, length in self._free)

    def can_allocate(self, byte_count: int) -> bool:
        required = _align_up(byte_count, self.alignment)
        return any(length >= required for _, length in self._free)

    def allocate(self, byte_count: int) -> tuple[int, int]:
        required = _align_up(byte_count, self.alignment)
        if required == 0:
            raise HBServeError("cannot allocate an empty extent")
        for index, (begin, length) in enumerate(self._free):
            if length < required:
                continue
            allocation = (begin, required)
            if length == required:
                del self._free[index]
            else:
                self._free[index] = (begin + required, length - required)
            return allocation
        raise HBServeError(
            f"placement allocator cannot fit {required} bytes"
        )

    def release(self, begin: int, byte_count: int) -> None:
        required = _align_up(byte_count, self.alignment)
        if (
            required == 0
            or begin < self.begin
            or begin % self.alignment
            or begin + required > self.end
        ):
            raise HBServeError("released extent is outside the allocator")
        self._free.append((begin, required))
        self._free.sort()
        merged: list[tuple[int, int]] = []
        for cursor, length in self._free:
            if merged and merged[-1][0] + merged[-1][1] == cursor:
                prior_begin, prior_length = merged[-1]
                merged[-1] = (prior_begin, prior_length + length)
            else:
                if merged and merged[-1][0] + merged[-1][1] > cursor:
                    raise HBServeError(
                        "released extent overlaps existing free capacity"
                    )
                merged.append((cursor, length))
        self._free = merged

    def canonical(self) -> list[dict[str, int]]:
        return [
            {"begin": begin, "bytes": byte_count}
            for begin, byte_count in self._free
        ]


class _BlockPool:
    """Fixed-size KV block pool; lowest free block first (deterministic)."""

    def __init__(self, tier: str, begin: int, end: int, block_bytes: int) -> None:
        if begin < 0 or end < begin:
            raise HBServeError(f"{tier} KV pool range is invalid")
        self.tier = tier
        self.target = TIER_TARGETS[tier]
        self.begin = begin
        self.block_bytes = _positive_integer(block_bytes, "KV block bytes")
        self.capacity_blocks = (end - begin) // block_bytes
        self.end = begin + self.capacity_blocks * block_bytes
        self._next_unused = 0
        self._free_ranges: dict[int, int] = {}
        self._free_ends: dict[int, int] = {}
        self._free_starts: list[int] = []
        self.references = BlockReferences()

    @property
    def free_blocks(self) -> int:
        return self.capacity_blocks - len(self.references)

    @property
    def capacity_bytes(self) -> int:
        return self.capacity_blocks * self.block_bytes

    def allocate(self, count: int) -> list[int]:
        _nonnegative_integer(count, "KV allocation count")
        if count > self.free_blocks:
            raise HBServeError(
                f"{self.tier} KV pool cannot allocate {count} blocks "
                f"({self.free_blocks} free)"
            )
        blocks: list[int] = []
        while count and self._free_ranges:
            begin = heapq.heappop(self._free_starts)
            end = self._free_ranges.pop(begin, None)
            if end is None:
                continue
            del self._free_ends[end]
            take = min(count, end - begin)
            blocks.extend(range(begin, begin + take))
            if begin + take < end:
                self._free_ranges[begin + take] = end
                self._free_ends[end] = begin + take
                heapq.heappush(self._free_starts, begin + take)
            count -= take
        if count:
            blocks.extend(range(self._next_unused, self._next_unused + count))
            self._next_unused += count
        for block in blocks:
            self.references[block] = 1
        self._compact_free_index()
        return blocks

    def _compact_free_index(self) -> None:
        # Coalescing removes interval boundaries. Keep lazy heap tombstones
        # bounded even when the pool always has some live allocations.
        if len(self._free_starts) > 2 * len(self._free_ranges) + 64:
            self._free_starts = list(self._free_ranges)
            heapq.heapify(self._free_starts)
        if self._free_ranges.__sizeof__() > max(4096, 128 * len(self._free_ranges)):
            self._free_ranges = dict(self._free_ranges)
            self._free_ends = dict(self._free_ends)
        if not self._free_ranges:
            self._free_starts.clear()
            self._free_ranges.clear()
            self._free_ends.clear()

    def retain(self, blocks: Sequence[int]) -> None:
        for block in blocks:
            if block not in self.references:
                raise HBServeError("cannot retain an unallocated KV block")
            self.references[block] += 1

    def release(self, blocks: Sequence[int]) -> None:
        for block in blocks:
            if not 0 <= block < self.capacity_blocks:
                raise HBServeError(f"{self.tier} KV block {block} is out of range")
            if block not in self.references:
                raise HBServeError(f"{self.tier} KV pool released a block twice")
            references = self.references[block] - 1
            if references:
                self.references[block] = references
                continue
            del self.references[block]
            left = self._free_ends.pop(block, None)
            begin = block if left is None else left
            end = self._free_ranges.pop(block + 1, block + 1)
            if end != block + 1:
                del self._free_ends[end]
            if end == self._next_unused:
                self._next_unused = begin
                if left is not None:
                    del self._free_ranges[left]
            else:
                self._free_ranges[begin] = end
                self._free_ends[end] = begin
                if left is None:
                    heapq.heappush(self._free_starts, begin)
            self._compact_free_index()


    def address(self, block: int) -> int:
        if not 0 <= block < self.capacity_blocks:
            raise HBServeError(f"{self.tier} KV block {block} is out of range")
        return self.begin + block * self.block_bytes

    def canonical(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "target": self.target,
            "begin": self.begin,
            "block_bytes": self.block_bytes,
            "capacity_blocks": self.capacity_blocks,
            "capacity_bytes": self.capacity_bytes,
        }


@dataclass
class _RequestKv:
    request: RequestSpec
    model: ModelSpec
    order: int
    tier: str = KV_HOT_TIER
    blocks: list[list[int]] = field(default_factory=list)
    last_batch_id: int = -1
    prefix_keys: tuple[str, ...] = ()
    prefix_hit_tokens: int = 0

    @property
    def blocks_per_layer(self) -> int:
        return len(self.blocks[0]) if self.blocks else 0

    @property
    def total_blocks(self) -> int:
        return self.blocks_per_layer * self.model.num_layers


@dataclass(frozen=True)
class _Migration:
    kind: str  # swap_out | swap_in
    request_id: str
    pairs: tuple[tuple[tuple[int, int], ...], ...]  # per layer (source, destination)


@dataclass(frozen=True)
class _KvPlan:
    needed_hot_blocks: int
    swap_in: tuple[str, ...]
    victims: tuple[str, ...]
    prefix_evictions: tuple[str, ...] = ()


@dataclass(frozen=True)
class _CachedModel:
    model_id: str
    begin: int
    bytes: int
    objects: Mapping[str, ObjectAddress]


@dataclass(frozen=True)
class ModelActivation:
    model_id: str
    hit: bool
    evicted_models: tuple[str, ...]
    cached: _CachedModel | None


def _block_runs(
    pairs: Sequence[tuple[int, int]]
) -> list[tuple[int, int, int]]:
    """Merge consecutive (source, destination) pairs into (src, dst, count)."""

    runs: list[tuple[int, int, int]] = []
    for source, destination in pairs:
        if runs:
            run_source, run_destination, count = runs[-1]
            if (
                run_source + count == source
                and run_destination + count == destination
            ):
                runs[-1] = (run_source, run_destination, count + 1)
                continue
        runs.append((source, destination, 1))
    return runs


class HBServePlacement:
    """Stateful placement/remapping plane shared across HBServe batches."""

    def __init__(
        self,
        *,
        models: Mapping[str, ModelSpec],
        spec: PlacementSpec,
        hbf_geometry: HbfGeometry | None = None,
        hbm_stripe_bytes: int | None = None,
    ) -> None:
        if not models:
            raise HBServeError("serving placement requires models")
        if spec.prefix_cache_bytes and any(layer.is_moe for model in models.values() for layer in model.layers):
            raise HBServeError("MoE prefix reuse requires per-prefix routing identity; token-only prefix caching supports dense models only")
        self.models = dict(models)
        self.spec = spec
        self.hbf_geometry = hbf_geometry
        # The HBM address map rotates over every pseudo-channel once per
        # stripe.  Objects that start on a stripe boundary and pieces that
        # cover whole stripes give every pseudo-channel the same burst
        # sequence, which the device serves by symmetric replication instead
        # of burst by burst; the timing is bit-identical either way, so this
        # is a host-time property, and the split below conserves bytes.
        if hbm_stripe_bytes is not None:
            _positive_integer(hbm_stripe_bytes, "HBM stripe bytes")
        self.hbm_stripe_bytes = hbm_stripe_bytes
        self._hbm_object_alignment = (
            spec.hbm_alignment_bytes
            if hbm_stripe_bytes is None
            else math.lcm(spec.hbm_alignment_bytes, hbm_stripe_bytes)
        )
        if set(self.models) != set(spec.model_weight_tiers):
            raise HBServeError(
                "placement model coverage must exactly match the model catalog"
            )
        all_objects: dict[str, MemoryObject] = {}
        for model_id, model in self.models.items():
            if model_id != model.model_id:
                raise HBServeError(
                    "placement model key differs from ModelSpec model_id"
                )
            for memory_object in model.memory_objects:
                if memory_object.id in all_objects:
                    raise HBServeError(
                        f"duplicate global memory object {memory_object.id}"
                    )
                all_objects[memory_object.id] = memory_object
        unknown_overrides = set(spec.object_tier_overrides) - set(all_objects)
        if unknown_overrides:
            raise HBServeError(
                f"placement overrides unknown objects: {sorted(unknown_overrides)}"
            )
        self.objects = all_objects
        self._object_tier: dict[str, str] = {}
        for object_id, memory_object in self.objects.items():
            model_tier = spec.model_weight_tiers[memory_object.model_id]
            override = spec.object_tier_overrides.get(object_id)
            if model_tier == "external_cached_hbm" and override is not None:
                raise HBServeError(
                    "cached whole-model placement cannot have object overrides"
                )
            self._object_tier[object_id] = override or model_tier

        self._static: dict[str, ObjectAddress] = {}
        hbm_cursor = 0
        hbf_cursor = 0
        external_cursor = 0
        self._cached_model_offsets: dict[str, dict[str, int]] = {}
        self._cached_model_bytes: dict[str, int] = {}
        for model_id in sorted(self.models):
            model = self.models[model_id]
            cache_offset = 0
            cache_offsets: dict[str, int] = {}
            for memory_object in model.memory_objects:
                tier = self._object_tier[memory_object.id]
                if tier == "hbm":
                    hbm_cursor = self._align_hbm_object(
                        hbm_cursor, memory_object.bytes
                    )
                    self._static[memory_object.id] = ObjectAddress(
                        memory_object.id,
                        "HBM",
                        hbm_cursor,
                        memory_object.bytes,
                    )
                    hbm_cursor += memory_object.bytes
                elif tier == "hbf":
                    hbf_cursor = _align_up(
                        hbf_cursor, spec.hbf_page_size_bytes
                    )
                    self._static[memory_object.id] = ObjectAddress(
                        memory_object.id,
                        "HBF_LOGICAL",
                        hbf_cursor,
                        memory_object.bytes,
                    )
                    hbf_cursor += memory_object.bytes
                elif tier in {"external_direct", "external_cached_hbm"}:
                    external_cursor = _align_up(
                        external_cursor, spec.external_page_size_bytes
                    )
                    self._static[memory_object.id] = ObjectAddress(
                        memory_object.id,
                        "EXTERNAL",
                        external_cursor,
                        memory_object.bytes,
                    )
                    external_cursor += memory_object.bytes
                    if tier == "external_cached_hbm":
                        cache_offset = self._align_hbm_object(
                            cache_offset, memory_object.bytes
                        )
                        cache_offsets[memory_object.id] = cache_offset
                        cache_offset += memory_object.bytes
                else:  # Defensive: PlacementSpec already validates this.
                    raise HBServeError(f"unsupported tier {tier}")
            if spec.model_weight_tiers[model_id] == "external_cached_hbm":
                extent = self._align_hbm_object(cache_offset, cache_offset)
                self._cached_model_offsets[model_id] = cache_offsets
                self._cached_model_bytes[model_id] = extent
                if extent > spec.hbm_model_cache_bytes:
                    raise HBServeError(
                        f"cached model {model_id} ({extent} bytes) exceeds the "
                        "HBM model cache"
                    )

        hbm_static_end = self._align_hbm_object(hbm_cursor)
        self.hbm_cache_begin = hbm_static_end
        self.hbm_cache_end = self.hbm_cache_begin + spec.hbm_model_cache_bytes
        self.hbm_kv_begin = self._align_hbm_object(self.hbm_cache_end)
        self.hbm_kv_end = spec.hbm_capacity_bytes - spec.hbm_runtime_reserve_bytes
        if self.hbm_kv_end < self.hbm_kv_begin:
            raise HBServeError(
                "static HBM weights, model cache, and runtime reserve exceed HBM"
            )
        self.hbm_static_bytes = hbm_static_end
        self.hbf_allocation_bytes = _align_up(
            hbf_cursor, spec.hbf_page_size_bytes
        )
        self.external_allocation_bytes = _align_up(
            external_cursor, spec.external_page_size_bytes
        )
        if self.hbf_allocation_bytes > spec.hbf_capacity_bytes:
            raise HBServeError(
                "HBF-resident model objects exceed configured HBF payload capacity"
            )
        if self.external_allocation_bytes > spec.external_capacity_bytes:
            raise HBServeError(
                "external model objects exceed configured external capacity"
            )
        self._cache_allocator = _ExtentAllocator(
            self.hbm_cache_begin,
            self.hbm_cache_end,
            spec.hbm_alignment_bytes,
        )

        # One block size for every model and layer keeps the pools uniform;
        # a layer whose block payload is smaller than the stride leaves the
        # remainder unused and the receipt reports the utilization.
        block_alignment = spec.hbm_alignment_bytes
        cold_page = spec.kv_cold_page_bytes
        if cold_page is not None:
            block_alignment = max(block_alignment, cold_page)
        self.kv_block_bytes = max(
            _align_up(
                spec.kv_block_tokens * layer.kv_bytes_per_token, block_alignment
            )
            for model in self.models.values()
            for layer in model.layers
        )
        self._hot = _BlockPool(
            KV_HOT_TIER, self.hbm_kv_begin, self.hbm_kv_end, self.kv_block_bytes
        )
        if spec.prefix_cache_bytes > self._hot.capacity_bytes:
            raise HBServeError("prefix cache budget exceeds the shared HBM KV pool")
        self._prefix = PrefixCache(self._hot, spec.prefix_cache_bytes, spec.prefix_cache_ttl_ns)
        self._model_digests = {model_id: model.digest for model_id, model in self.models.items()}
        self._cold: _BlockPool | None = None
        cold_tier = spec.kv_placement.cold
        if cold_tier == "hbf":
            if hbf_geometry is None:
                raise HBServeError(
                    "cold KV in HBF requires the HBF geometry for D2D link "
                    "decomposition"
                )
            self._cold = _BlockPool(
                "hbf",
                self.hbf_allocation_bytes,
                spec.hbf_capacity_bytes,
                self.kv_block_bytes,
            )
        elif cold_tier == "external":
            self._cold = _BlockPool(
                "external",
                self.external_allocation_bytes,
                spec.external_capacity_bytes,
                self.kv_block_bytes,
            )
        if self._cold is not None and self._cold.capacity_blocks == 0:
            raise HBServeError("cold KV tier has no block capacity")

        self._cache: OrderedDict[str, _CachedModel] = OrderedDict()
        self._kv: dict[str, _RequestKv] = {}
        self._next_order = 0
        self._pending_migrations: list[_Migration] = []
        self._cold_block_fences: dict[int, str] = {}
        self._pending_batch_id: int | None = None
        self._seen_batches: set[int] = set()
        self._activation_count = 0
        self._eviction_count = 0
        self._swap_out_blocks = 0
        self._swap_in_blocks = 0
        self._preemptions = 0

        for model_id in spec.initial_cached_models:
            if spec.model_weight_tiers.get(model_id) != "external_cached_hbm":
                raise HBServeError(
                    "initial cached models must use external_cached_hbm"
                )
            activation = self._activate_model(model_id)
            if activation.hit:
                raise HBServeError("duplicate initial cached model")
        # Setup population is not a serving-time activation or eviction.
        self._activation_count = 0
        self._eviction_count = 0

    # ------------------------------------------------------------ HBM map
    def _align_hbm_object(self, value: int, byte_count: int | None = None) -> int:
        """Round up to the HBM allocation granularity, and to the stripe when
        the object is at least one stripe long (shorter objects can never be
        replicated, so stripe padding would only waste capacity)."""

        alignment = self._hbm_object_alignment
        if (
            byte_count is not None
            and self.hbm_stripe_bytes is not None
            and byte_count < self.hbm_stripe_bytes
        ):
            alignment = self.spec.hbm_alignment_bytes
        return (value + alignment - 1) // alignment * alignment

    def _hbm_stripe_pieces(
        self, addr: int, byte_count: int
    ) -> list[tuple[int, int]]:
        """Split one HBM extent into a whole-stripe head and scalar remainders.

        ``[addr, first)`` and ``[last, end)`` are the partial stripes at either
        end; ``[first, last)`` covers whole stripes and is the piece the HBM
        engine can replicate.  Bytes are conserved exactly.
        """

        stripe = self.hbm_stripe_bytes
        if stripe is None or byte_count < stripe:
            return [(addr, byte_count)]
        end = addr + byte_count
        first = (addr + stripe - 1) // stripe * stripe
        last = end // stripe * stripe
        if last <= first:
            return [(addr, byte_count)]
        pieces: list[tuple[int, int]] = []
        if first > addr:
            pieces.append((addr, first - addr))
        pieces.append((first, last - first))
        if end > last:
            pieces.append((last, end - last))
        return pieces

    # ----------------------------------------------------------------- tiers
    @property
    def enable_hbf(self) -> bool:
        return self.hbf_allocation_bytes > 0 or self.spec.kv_placement.cold == "hbf"

    @property
    def enable_external(self) -> bool:
        return (
            self.external_allocation_bytes > 0
            or self.spec.kv_placement.cold == "external"
        )

    @property
    def initial_hbf_logical_pages(self) -> int:
        return self.hbf_allocation_bytes // self.spec.hbf_page_size_bytes

    def _state_receipt(self) -> dict[str, Any]:
        return {
            "cache_lru": [
                {
                    "model_id": model_id,
                    "begin": entry.begin,
                    "bytes": entry.bytes,
                }
                for model_id, entry in self._cache.items()
            ],
            "live_kv": [
                {
                    "request_id": request_id,
                    "tier": state.tier,
                    "blocks_per_layer": state.blocks_per_layer,
                    "tokens_covered": (
                        state.blocks_per_layer * self.spec.kv_block_tokens
                    ),
                    "last_batch_id": state.last_batch_id,
                }
                for request_id, state in sorted(self._kv.items())
            ],
            "cache_free_extents": self._cache_allocator.canonical(),
            "kv_hot_free_blocks": self._hot.free_blocks,
            "prefix_cache": self._prefix.receipt(),
            "kv_cold_free_blocks": (
                None if self._cold is None else self._cold.free_blocks
            ),
        }

    @property
    def frontier_ns_independent_state(self) -> dict[str, Any]:
        return {
            "placement_sha256": self.spec.digest,
            "hbm_static_bytes": self.hbm_static_bytes,
            "hbm_model_cache": {
                "begin": self.hbm_cache_begin,
                "bytes": self.spec.hbm_model_cache_bytes,
            },
            "kv": {
                "allocation_policy": KV_ALLOCATION_POLICY,
                "migration_policy": KV_MIGRATION_POLICY,
                "block_tokens": self.spec.kv_block_tokens,
                "block_bytes": self.kv_block_bytes,
                "hot": self._hot.canonical(),
                "cold": None if self._cold is None else self._cold.canonical(),
            },
            "hbm_runtime_reserve_bytes": self.spec.hbm_runtime_reserve_bytes,
            "hbm_stripe_bytes": self.hbm_stripe_bytes,
            "hbm_object_alignment_bytes": self._hbm_object_alignment,
            "hbf_allocation_bytes": self.hbf_allocation_bytes,
            "external_allocation_bytes": self.external_allocation_bytes,
            "initial_cached_models": list(self.spec.initial_cached_models),
        }

    # -------------------------------------------------------------- requests
    def admit_request(self, request: RequestSpec, *, now_ns: float = 0.0) -> int:
        """Register a request; blocks are allocated later by ``reserve``."""

        if request.request_id in self._kv:
            return self._kv[request.request_id].prefix_hit_tokens
        try:
            model = self.models[request.model_id]
        except KeyError as error:
            raise HBServeError(
                f"cannot admit request for unknown model {request.model_id}"
            ) from error
        state = _RequestKv(
            request=request, model=model, order=self._next_order
        )
        if self.spec.prefix_cache_bytes and request.token_ids is not None:
            state.prefix_keys = block_keys(request, self._model_digests[model.model_id], self.spec.kv_block_tokens)
            entries = self._prefix.lookup(state.prefix_keys, (request.prompt_tokens - 1) // self.spec.kv_block_tokens,
                                          self.spec.kv_block_tokens, now_ns)
            if entries:
                state.blocks = [[entry[layer] for entry in entries] for layer in range(model.num_layers)]
                state.prefix_hit_tokens = len(entries) * self.spec.kv_block_tokens
        self._kv[request.request_id] = state
        self._next_order += 1
        return state.prefix_hit_tokens

    def complete_batch(self, batch: CanonicalServingBatch, finish_ns: float) -> None:
        if not self.spec.prefix_cache_bytes:
            return
        for item in batch.schedule.slices:
            state = self._state(item.request_id)
            self._prefix.publish(state.prefix_keys, state.blocks,
                                 min(item.token_end, state.request.prompt_tokens) // self.spec.kv_block_tokens, finish_ns)

    def is_admitted(self, request_id: str) -> bool:
        return request_id in self._kv

    def _state(self, request_id: str) -> _RequestKv:
        try:
            return self._kv[request_id]
        except KeyError as error:
            raise HBServeError(
                f"request {request_id} has no live KV state"
            ) from error

    def _blocks_per_layer_for(self, tokens: int) -> int:
        return (tokens + self.spec.kv_block_tokens - 1) // self.spec.kv_block_tokens

    def _plan(self, slices: Sequence[BatchSlice]) -> _KvPlan | None:
        in_batch = {item.request_id for item in slices}
        if len(in_batch) != len(slices):
            raise HBServeError("KV plan received a repeated request")
        needed = 0
        swap_in: list[str] = []
        for item in slices:
            state = self._state(item.request_id)
            grow = max(
                0,
                self._blocks_per_layer_for(item.token_end) - state.blocks_per_layer,
            )
            needed += grow * state.model.num_layers
            if state.tier != KV_HOT_TIER:
                swap_in.append(item.request_id)
                needed += state.total_blocks
        victims: list[str] = []
        deficit = needed - self._hot.free_blocks
        prefix_evictions = []
        releases: dict[int, int] = {}

        def reclaimed(blocks):
            count = 0
            for block in blocks:
                releases[block] = releases.get(block, 0) + 1
                count += int(releases[block] == self._hot.references[block])
            return count

        if deficit > 0:
            for key, entry in self._prefix.entries.items():
                prefix_evictions.append(key)
                deficit -= reclaimed(entry.blocks)
                if deficit <= 0:
                    break
        if deficit > 0:
            if self._cold is None:
                return None
            cold_free = self._cold.free_blocks
            candidates = sorted(
                (
                    state
                    for state in self._kv.values()
                    if state.tier == KV_HOT_TIER
                    and state.blocks_per_layer
                    and state.request.request_id not in in_batch
                ),
                key=lambda state: (state.last_batch_id, state.order),
            )
            for state in candidates:
                if deficit <= 0:
                    break
                if state.total_blocks > cold_free:
                    return None
                victims.append(state.request.request_id)
                cold_free -= state.total_blocks
                deficit -= reclaimed(block for layer in state.blocks for block in layer)
            if deficit > 0:
                return None
        return _KvPlan(
            needed_hot_blocks=needed,
            swap_in=tuple(swap_in),
            victims=tuple(victims),
            prefix_evictions=tuple(prefix_evictions),
        )

    def can_reserve(self, slices: Sequence[BatchSlice]) -> bool:
        """True when the iteration's KV fits after migrating waiting requests."""

        return self._plan(slices) is not None

    def _migrate(self, state: _RequestKv, destination: _BlockPool, kind: str) -> int:
        source = self._hot if state.tier == KV_HOT_TIER else self._cold
        assert source is not None
        if source is destination:
            raise HBServeError("KV migration source and destination coincide")
        pairs: list[tuple[tuple[int, int], ...]] = []
        new_blocks: list[list[int]] = []
        for layer_blocks in state.blocks:
            destination_blocks = destination.allocate(len(layer_blocks))
            pairs.append(tuple(zip(layer_blocks, destination_blocks)))
            new_blocks.append(destination_blocks)
            source.release(layer_blocks)
        moved = state.total_blocks
        state.blocks = new_blocks
        state.tier = destination.tier
        self._pending_migrations.append(
            _Migration(
                kind=kind, request_id=state.request.request_id, pairs=tuple(pairs)
            )
        )
        return moved

    def reserve(self, batch_id: int, slices: Sequence[BatchSlice]) -> dict[str, Any]:
        """Evict cold-eligible waiting requests, swap in, and grow block tables."""

        _nonnegative_integer(batch_id, "reserve batch id")
        if self._pending_batch_id is not None and self._pending_batch_id != batch_id:
            raise HBServeError("KV reservation for a different batch is pending")
        plan = self._plan(slices)
        if plan is None:
            raise HBServeError(
                "HBM KV pool cannot hold the iteration even after migrating "
                "every waiting request"
            )
        swap_out_blocks = 0
        for key in plan.prefix_evictions:
            self._prefix.evict(key)
        for request_id in plan.victims:
            assert self._cold is not None
            swap_out_blocks += self._migrate(
                self._state(request_id), self._cold, "swap_out"
            )
        swap_in_blocks = 0
        for request_id in plan.swap_in:
            swap_in_blocks += self._migrate(
                self._state(request_id), self._hot, "swap_in"
            )
        allocated = 0
        for item in slices:
            state = self._state(item.request_id)
            grow = max(
                0,
                self._blocks_per_layer_for(item.token_end) - state.blocks_per_layer,
            )
            if grow:
                if not state.blocks:
                    state.blocks = [[] for _ in range(state.model.num_layers)]
                for layer_blocks in state.blocks:
                    layer_blocks.extend(self._hot.allocate(grow))
                allocated += grow * state.model.num_layers
            state.last_batch_id = batch_id
        self._pending_batch_id = batch_id
        self._swap_out_blocks += swap_out_blocks
        self._swap_in_blocks += swap_in_blocks
        return {
            "batch_id": batch_id,
            "allocated_blocks": allocated,
            "swap_out_requests": list(plan.victims),
            "swap_out_blocks": swap_out_blocks,
            "swap_in_requests": list(plan.swap_in),
            "swap_in_blocks": swap_in_blocks,
            "hot_free_blocks_after": self._hot.free_blocks,
            "prefix_cache_evictions": len(plan.prefix_evictions),
        }

    def preempt_request(self, request_id: str) -> dict[str, Any]:
        """Take a running request's KV out of HBM.

        With a cold tier that has room the blocks migrate whole-request
        (``swap_out``) and the request resumes by swapping back in; otherwise
        the blocks are freed and the request must recompute its context.
        """

        state = self._state(request_id)
        blocks = state.total_blocks
        if state.tier != KV_HOT_TIER or blocks == 0:
            raise HBServeError(
                f"request {request_id} holds no hot KV blocks to preempt"
            )
        self._preemptions += 1
        if self._cold is not None and self._cold.free_blocks >= blocks:
            self._migrate(state, self._cold, "swap_out")
            self._swap_out_blocks += blocks
            mode = "swap_out"
        else:
            for layer_blocks in state.blocks:
                self._hot.release(layer_blocks)
            state.blocks = []
            mode = "recompute"
        return {
            "request_id": request_id,
            "mode": mode,
            "blocks": blocks,
            "bytes": blocks * self.kv_block_bytes,
        }

    def release_request(self, request_id: str) -> None:
        state = self._kv.pop(request_id, None)
        if state is None:
            raise HBServeError(
                f"request {request_id} has no live KV allocation"
            )
        pool = self._hot if state.tier == KV_HOT_TIER else self._cold
        assert pool is not None
        for layer_blocks in state.blocks:
            pool.release(layer_blocks)

    # ----------------------------------------------------------- weights
    def _cached_entry(self, model_id: str, begin: int, extent: int) -> _CachedModel:
        offsets = self._cached_model_offsets[model_id]
        objects = {
            object_id: ObjectAddress(
                object_id=object_id,
                target="HBM",
                addr=begin + offset,
                bytes=self.objects[object_id].bytes,
            )
            for object_id, offset in offsets.items()
        }
        return _CachedModel(
            model_id=model_id,
            begin=begin,
            bytes=extent,
            objects=objects,
        )

    def _activate_model(self, model_id: str) -> ModelActivation:
        if self.spec.model_weight_tiers[model_id] != "external_cached_hbm":
            return ModelActivation(model_id, True, (), None)
        existing = self._cache.pop(model_id, None)
        if existing is not None:
            self._cache[model_id] = existing
            return ModelActivation(model_id, True, (), existing)
        extent = self._cached_model_bytes[model_id]
        evicted: list[str] = []
        while not self._cache_allocator.can_allocate(extent):
            if not self._cache:
                raise HBServeError(
                    f"HBM model cache cannot fit model {model_id}"
                )
            victim_id, victim = self._cache.popitem(last=False)
            self._cache_allocator.release(victim.begin, victim.bytes)
            evicted.append(victim_id)
        begin, allocated = self._cache_allocator.allocate(extent)
        entry = self._cached_entry(model_id, begin, allocated)
        self._cache[model_id] = entry
        self._activation_count += 1
        self._eviction_count += len(evicted)
        return ModelActivation(model_id, False, tuple(evicted), entry)

    def _resolve_weight(self, object_id: str) -> ObjectAddress:
        try:
            tier = self._object_tier[object_id]
        except KeyError as error:
            raise HBServeError(
                f"placement cannot resolve object {object_id}"
            ) from error
        if tier != "external_cached_hbm":
            return self._static[object_id]
        model_id = self.objects[object_id].model_id
        try:
            return self._cache[model_id].objects[object_id]
        except KeyError as error:
            raise HBServeError(
                f"cached model {model_id} is not activated"
            ) from error

    def _kv_pieces(
        self, object_id: str, offset: int, byte_count: int
    ) -> list[tuple[int, int]]:
        """Split one logical KV extent into physically contiguous HBM pieces."""

        parts = object_id.split("/")
        # request/<id>/model/<model>/layer/<layer>/kv
        if len(parts) != 7 or parts[0] != "request" or parts[6] != "kv":
            raise HBServeError(f"malformed KV object ID {object_id}")
        state = self._state(parts[1])
        if state.tier != KV_HOT_TIER:
            raise HBServeError(
                f"KV of request {parts[1]} is not resident in HBM at mapping time"
            )
        layer = int(parts[5])
        if layer >= state.model.num_layers or state.model.model_id != parts[3]:
            raise HBServeError(f"KV object {object_id} names a foreign layer")
        used_block_bytes = (
            self.spec.kv_block_tokens * state.model.layers[layer].kv_bytes_per_token
        )
        blocks = state.blocks[layer] if state.blocks else []
        pieces: list[tuple[int, int]] = []
        cursor = offset
        end = offset + byte_count
        while cursor < end:
            ordinal, within = divmod(cursor, used_block_bytes)
            take = min(used_block_bytes - within, end - cursor)
            if ordinal >= len(blocks):
                raise HBServeError(
                    f"KV access to {object_id} exceeds its allocated blocks"
                )
            addr = self._hot.address(blocks[ordinal]) + within
            if pieces and pieces[-1][0] + pieces[-1][1] == addr:
                pieces[-1] = (pieces[-1][0], pieces[-1][1] + take)
            else:
                pieces.append((addr, take))
            cursor += take
        return pieces

    # ------------------------------------------------------------ mapping
    def map_batch(
        self,
        batch: CanonicalServingBatch,
        *,
        session_frontier_ns: float,
    ) -> TransactionBatch:
        batch_id = batch.schedule.batch_id
        if batch_id in self._seen_batches:
            raise HBServeError("placement saw a duplicate batch ID")
        if (
            isinstance(session_frontier_ns, bool)
            or not isinstance(session_frontier_ns, (int, float))
            or not math.isfinite(float(session_frontier_ns))
            or session_frontier_ns < 0
        ):
            raise HBServeError("session frontier must be finite and >= 0")
        if self._pending_batch_id is not None and self._pending_batch_id != batch_id:
            raise HBServeError(
                "mapped batch differs from the batch whose KV was reserved"
            )
        for batch_slice in batch.schedule.slices:
            state = self._state(batch_slice.request_id)
            if (
                state.tier != KV_HOT_TIER
                or state.blocks_per_layer * self.spec.kv_block_tokens
                < batch_slice.token_end
            ):
                raise HBServeError(
                    f"request {batch_slice.request_id} has no reserved hot KV "
                    "blocks for this iteration"
                )

        activation = self._activate_model(batch.schedule.model_id)
        transactions: list[Transaction] = []
        projection: dict[str, tuple[str, ...]] = {}
        terminal: dict[str, str] = {}
        compute_windows: list[dict[str, Any]] = []
        phases = {item.phase for item in batch.schedule.slices}
        compute_phase = next(iter(phases)) if len(phases) == 1 else "mixed"
        counter = 0

        def emit(
            *,
            target: str,
            op: str | None,
            addr: int,
            byte_count: int,
            dependencies: Sequence[str],
            duration_ns: float = 0.0,
            stack: int | None = None,
        ) -> str:
            nonlocal counter
            identifier = f"mapped/b{batch_id}/p{counter}"
            counter += 1
            transactions.append(
                Transaction(
                    id=identifier,
                    target=target,
                    op=op,
                    addr=addr,
                    bytes=byte_count,
                    issue_ns=0.0,
                    duration_ns=duration_ns,
                    dependencies=tuple(dict.fromkeys(dependencies)),
                    stack=stack,
                )
            )
            return identifier

        idle_ns = max(0.0, batch.schedule.not_before_ns - session_frontier_ns)
        release = emit(
            target="BARRIER",
            op=None,
            addr=0,
            byte_count=0,
            dependencies=(),
            duration_ns=idle_ns,
        )
        roots = [release]

        # --- whole-model activation into the HBM cache ---------------------
        activation_source_bytes = 0
        activation_hbm_write_bytes = 0
        activation_chunks = 0
        if not activation.hit:
            if activation.cached is None:
                raise HBServeError("model activation has no cache extent")
            installed: list[str] = []
            for memory_object in self.models[
                batch.schedule.model_id
            ].memory_objects:
                source = self._static[memory_object.id]
                destination = activation.cached.objects[memory_object.id]
                if source.target != "EXTERNAL" or destination.target != "HBM":
                    raise HBServeError(
                        "cached model activation source/destination is malformed"
                    )
                for offset in range(
                    0, source.bytes, self.spec.model_load_chunk_bytes
                ):
                    chunk_bytes = min(
                        self.spec.model_load_chunk_bytes,
                        source.bytes - offset,
                    )
                    fetched = emit(
                        target="EXTERNAL",
                        op="R",
                        addr=source.addr + offset,
                        byte_count=chunk_bytes,
                        dependencies=(release,),
                    )
                    installed.append(
                        emit(
                            target="HBM",
                            op="W",
                            addr=destination.addr + offset,
                            byte_count=chunk_bytes,
                            dependencies=(fetched,),
                        )
                    )
                    activation_source_bytes += chunk_bytes
                    activation_hbm_write_bytes += chunk_bytes
                    activation_chunks += 1
            roots.append(
                emit(
                    target="BARRIER",
                    op=None,
                    addr=0,
                    byte_count=0,
                    dependencies=installed,
                )
            )

        # --- KV migrations: swap-outs free HBM, swap-ins fill it ----------
        migration_summary = {
            "swap_out": {"requests": 0, "blocks": 0, "bytes": 0},
            "swap_in": {"requests": 0, "blocks": 0, "bytes": 0},
        }
        migration_traffic: dict[str, dict[str, int]] = {}

        def account_migration(target: str, op: str, byte_count: int) -> None:
            row = migration_traffic.setdefault(
                target, {"operations": 0, "read_bytes": 0, "write_bytes": 0}
            )
            row["operations"] += 1
            row["read_bytes" if op == "R" else "write_bytes"] += byte_count

        def cold_links(
            *, target: str, op: str, addr: int, byte_count: int, dependency: str
        ) -> list[str]:
            assert self.hbf_geometry is not None
            links: list[str] = []
            for stack, stack_bytes in enumerate(
                hbf_link_bytes_by_stack(addr, byte_count, self.hbf_geometry)
            ):
                if stack_bytes:
                    links.append(
                        emit(
                            target=target,
                            op=op,
                            addr=addr,
                            byte_count=stack_bytes,
                            dependencies=(dependency,),
                            stack=stack,
                        )
                    )
                    account_migration(target, op, stack_bytes)
            return links

        swap_out_reads: list[str] = []
        swap_in_writes: list[str] = []
        for migration in self._pending_migrations:
            assert self._cold is not None
            summary = migration_summary[migration.kind]
            summary["requests"] += 1
            for layer_pairs in migration.pairs:
                for source_block, destination_block, count in _block_runs(
                    layer_pairs
                ):
                    byte_count = count * self.kv_block_bytes
                    summary["blocks"] += count
                    summary["bytes"] += byte_count
                    if migration.kind == "swap_out":
                        hot_addr = self._hot.address(source_block)
                        cold_addr = self._cold.address(destination_block)
                        cold_blocks = range(destination_block, destination_block + count)
                        cold_dependencies = tuple(dict.fromkeys(
                            self._cold_block_fences[block]
                            for block in cold_blocks
                            if block in self._cold_block_fences
                        ))
                        read = emit(
                            target="HBM",
                            op="R",
                            addr=hot_addr,
                            byte_count=byte_count,
                            dependencies=(release,),
                        )
                        account_migration("HBM", "R", byte_count)
                        swap_out_reads.append(read)
                        if self._cold.tier == "hbf":
                            links = cold_links(
                                target="D2D_HBM_TO_HBF",
                                op="W",
                                addr=cold_addr,
                                byte_count=byte_count,
                                dependency=read,
                            )
                            write = emit(
                                target="HBF_LOGICAL",
                                op="W",
                                addr=cold_addr,
                                byte_count=byte_count,
                                dependencies=(*links, *cold_dependencies),
                            )
                            account_migration("HBF_LOGICAL", "W", byte_count)
                        else:
                            write = emit(
                                target="EXTERNAL",
                                op="W",
                                addr=cold_addr,
                                byte_count=byte_count,
                                dependencies=(read, *cold_dependencies),
                            )
                            account_migration("EXTERNAL", "W", byte_count)
                        for block in cold_blocks:
                            self._cold_block_fences[block] = write
                    else:
                        cold_addr = self._cold.address(source_block)
                        hot_addr = self._hot.address(destination_block)
                        cold_blocks = range(source_block, source_block + count)
                        cold_dependencies = tuple(dict.fromkeys(
                            self._cold_block_fences[block]
                            for block in cold_blocks
                            if block in self._cold_block_fences
                        ))
                        # Swap-ins reuse HBM blocks that swap-outs may have
                        # just released, so they wait for those reads.
                        fetch_dependencies = [
                            release, *swap_out_reads, *cold_dependencies
                        ]
                        if self._cold.tier == "hbf":
                            read = emit(
                                target="HBF_LOGICAL",
                                op="R",
                                addr=cold_addr,
                                byte_count=byte_count,
                                dependencies=fetch_dependencies,
                            )
                            account_migration("HBF_LOGICAL", "R", byte_count)
                            arrival = cold_links(
                                target="D2D_HBF_TO_HBM",
                                op="R",
                                addr=cold_addr,
                                byte_count=byte_count,
                                dependency=read,
                            )
                        else:
                            read = emit(
                                target="EXTERNAL",
                                op="R",
                                addr=cold_addr,
                                byte_count=byte_count,
                                dependencies=fetch_dependencies,
                            )
                            arrival = [read]
                            account_migration("EXTERNAL", "R", byte_count)
                        for block in cold_blocks:
                            self._cold_block_fences[block] = read
                        swap_in_writes.append(
                            emit(
                                target="HBM",
                                op="W",
                                addr=hot_addr,
                                byte_count=byte_count,
                                dependencies=arrival,
                            )
                        )
                        account_migration("HBM", "W", byte_count)
        if swap_out_reads or swap_in_writes:
            roots.append(
                emit(
                    target="BARRIER",
                    op=None,
                    addr=0,
                    byte_count=0,
                    dependencies=(*swap_out_reads, *swap_in_writes),
                )
            )
        self._pending_migrations = []
        self._pending_batch_id = None

        # --- canonical operations --------------------------------------
        logical_bytes = 0
        logical_by_target: dict[str, dict[str, int]] = {}
        for operation in batch.operations:
            dependencies = [terminal[item] for item in operation.dependencies]
            if not dependencies:
                dependencies.extend(roots)
            if operation.is_barrier:
                terminal[operation.id] = emit(
                    target="BARRIER",
                    op=None,
                    addr=0,
                    byte_count=0,
                    dependencies=dependencies,
                    duration_ns=operation.duration_ns,
                )
                if operation.role.endswith(("/compute", "/routing_ready")) and operation.duration_ns > 0:
                    compute_windows.append({
                        "transaction_id": terminal[operation.id],
                        "operation_id": operation.id,
                        "role": operation.role,
                        "duration_ns": operation.duration_ns,
                        "phase": compute_phase,
                    })
                continue
            assert operation.object_id is not None and operation.op is not None
            if operation.object_id.startswith("request/"):
                pieces = self._kv_pieces(
                    operation.object_id, operation.offset, operation.bytes
                )
                target = "HBM"
            else:
                placed = self._resolve_weight(operation.object_id)
                if operation.offset + operation.bytes > placed.bytes:
                    raise HBServeError(
                        f"mapped access escapes object {operation.object_id}"
                    )
                pieces = [(placed.addr + operation.offset, operation.bytes)]
                target = placed.target
            if target == "HBM":
                pieces = [
                    part
                    for piece in pieces
                    for part in self._hbm_stripe_pieces(*piece)
                ]
            mapped_ids = [
                emit(
                    target=target,
                    op=operation.op,
                    addr=addr,
                    byte_count=byte_count,
                    dependencies=dependencies,
                )
                for addr, byte_count in pieces
            ]
            terminal[operation.id] = (
                mapped_ids[0]
                if len(mapped_ids) == 1
                else emit(
                    target="BARRIER",
                    op=None,
                    addr=0,
                    byte_count=0,
                    dependencies=mapped_ids,
                )
            )
            projection[operation.id] = tuple(mapped_ids)
            projected_bytes = sum(byte_count for _, byte_count in pieces)
            if projected_bytes != operation.bytes:
                raise HBServeError(
                    f"mapped pieces of {operation.id} do not conserve its bytes"
                )
            logical_bytes += projected_bytes
            target_row = logical_by_target.setdefault(
                target,
                {"operations": 0, "read_bytes": 0, "write_bytes": 0},
            )
            target_row["operations"] += len(mapped_ids)
            target_row[
                "read_bytes" if operation.op == "R" else "write_bytes"
            ] += operation.bytes
        if logical_bytes != batch.logical_bytes or set(projection) != {
            operation.id for operation in batch.memory_operations
        }:
            raise HBServeError(
                "serving remap does not conserve canonical memory operations"
            )

        state = self._state_receipt()
        receipt = {
            "schema": REMAP_SCHEMA,
            "result": "pass",
            "batch_id": batch_id,
            # Explicit frontend information for memory policies. Arrival waits
            # and synchronization barriers are never compute opportunities.
            "compute_windows": compute_windows,
            "source": {
                "canonical_batch_sha256": batch.digest,
                "placement_sha256": self.spec.digest,
            },
            "activation": {
                "model_id": activation.model_id,
                "cache_hit": activation.hit,
                "evicted_models": list(activation.evicted_models),
                "external_read_bytes": activation_source_bytes,
                "hbm_install_write_bytes": activation_hbm_write_bytes,
                "chunks": activation_chunks,
                "chunk_bytes": self.spec.model_load_chunk_bytes,
            },
            "kv_migration": {
                **migration_summary,
                "traffic_by_target": {
                    target: migration_traffic[target]
                    for target in sorted(migration_traffic)
                },
            },
            "invariants": {
                "canonical_memory_operations": len(batch.memory_operations),
                "canonical_logical_bytes": batch.logical_bytes,
                "projected_memory_operations": len(projection),
                "projected_transactions": sum(
                    len(ids) for ids in projection.values()
                ),
                "projected_logical_bytes": logical_bytes,
                "canonical_bytes_conserved": True,
                "dependency_order_preserved": True,
                "policy_overhead_separated": True,
                "semantic_fields_at_execution_boundary": False,
            },
            "policy_overhead": {
                "external_model_load_bytes": activation_source_bytes,
                "hbm_model_install_bytes": activation_hbm_write_bytes,
                "model_load_chunks": activation_chunks,
                "kv_swap_out_bytes": migration_summary["swap_out"]["bytes"],
                "kv_swap_in_bytes": migration_summary["swap_in"]["bytes"],
            },
            "logical_traffic_by_target": {
                target: logical_by_target[target]
                for target in sorted(logical_by_target)
            },
            "state_after": state,
            "cumulative": self._cumulative(),
        }
        routing_digest = canonical_sha256(
            {
                "placement_sha256": self.spec.digest,
                "batch_id": batch_id,
                "activation": receipt["activation"],
                "kv_migration": receipt["kv_migration"],
                "state_after": state,
            }
        )
        self._seen_batches.add(batch_id)
        return TransactionBatch(
            batch_id=batch_id,
            logical_trace_sha256=batch.digest,
            routing_sidecar_sha256=routing_digest,
            transactions=tuple(transactions),
            retain=tuple(sorted(set(self._cold_block_fences.values()))),
            receipt=receipt,
        )

    def _cumulative(self) -> dict[str, int]:
        return {
            "model_activations": self._activation_count,
            "model_evictions": self._eviction_count,
            "kv_swap_out_blocks": self._swap_out_blocks,
            "kv_swap_in_blocks": self._swap_in_blocks,
            "kv_swap_out_bytes": self._swap_out_blocks * self.kv_block_bytes,
            "kv_swap_in_bytes": self._swap_in_blocks * self.kv_block_bytes,
            "preemptions": self._preemptions,
        }

    def receipt(self) -> dict[str, Any]:
        state = self._state_receipt()
        return {
            "schema": PLACEMENT_SCHEMA,
            "result": "pass",
            "spec": self.spec.canonical(),
            "layout": deepcopy(self.frontier_ns_independent_state),
            "static_objects": [
                self._static[object_id].canonical()
                for object_id in sorted(self._static)
            ],
            "cumulative": self._cumulative(),
            "final_state": state,
            "prefix_cache": self._prefix.receipt(),
            "final_invariants": {
                "all_request_kv_released": not self._kv,
                "kv_hot_pool_fully_free": (
                    self._hot.free_blocks == self._hot.capacity_blocks
                ),
                "kv_hot_allocations_accounted_for": self._hot.free_blocks + len(self._hot.references) == self._hot.capacity_blocks,
                "kv_cold_pool_fully_free": (
                    self._cold is None
                    or self._cold.free_blocks == self._cold.capacity_blocks
                ),
                "no_pending_migrations": not self._pending_migrations,
            },
        }
