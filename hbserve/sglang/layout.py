"""Capacity-accounted homes for native slots; placement never allocates KV IDs.

Priority reserves the configured KV pool (including its native reserved page),
not the instantaneous live set. Homes are fixed for the run, without migration.
All layers use the same HBM slot cutoff. HBF weights occupy a prefix so the
physical engine can install exactly those immutable pages before serving.
"""
from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right
import math

from .model import positive_int


def align(n, unit):
    return (n + unit - 1) // unit * unit


@dataclass(frozen=True)
class Extent:
    offset: int
    size: int
    tier: str
    address: int


class Placement:
    def __init__(self, model, system, config, *, enable_hbf, enable_external):
        page = system.hbf_geometry.page_size_bytes
        self.page = page
        native_page = positive_int(config.get("page_size", 1), "page_size")
        self.kv_bytes = model.layers[0].kv_bytes_per_token
        layers = model.num_layers
        capacities = {"hbm": system.integer("hbm-capacity-bytes"),
                      "hbf": system.logical_hbf_capacity_bytes or 0,
                      "external": int(system.external_backing_identity["capacity_bytes"]) if enable_external else 0}
        controller = align(system.hbf_ctrl_dram_bytes, page) if enable_hbf else 0
        workspace = align(positive_int(config.get("workspace_bytes", 1048576), "workspace_bytes", 0), page)
        cache = positive_int(config.get("kv_cache_bytes", 0), "kv_cache_bytes", 0)
        priority = config.get("hbm_priority")
        weight_tier, kv_tier = config.get("weight_tier") or "hbf", config.get("kv_tier") or "hbf"
        chunk = config.get("transfer_chunk_bytes", 65536)
        staging = 2 * chunk if config.get("architecture", "tiered") == "tiered" and (
            enable_hbf or enable_external) else 0
        reserve = controller + workspace + cache + staging
        hbm_free = capacities["hbm"] - reserve
        if hbm_free < 0:
            raise ValueError("HBM workspace, controller and staging exceed capacity")
        objects = model.memory_objects
        weights = sum(align(obj.bytes, page) for obj in objects)
        # The partition must be native-page and physical-page aligned in every
        # layer; this also makes cache invalidation unambiguous at the boundary.
        slot_unit = math.lcm(native_page, page // math.gcd(page, self.kv_bytes))
        requested = config.get("max_total_tokens")
        if priority:
            if priority not in {"weights-first", "kv-first"}:
                raise ValueError("hbm_priority must be weights-first or kv-first")
            if requested is None:
                raise ValueError("priority placement requires explicit max_total_tokens")
            if config.get("weight_tier") or config.get("kv_tier"):
                raise ValueError("hbm_priority and explicit weight_tier/kv_tier are mutually exclusive")
            positive_int(requested, "max_total_tokens")
            pool = requested + native_page
            kv_hbm_slots = min(pool, (max(0, hbm_free - (weights if priority == "weights-first" else 0))
                                     // layers // self.kv_bytes) // slot_unit * slot_unit)
            kv_hbm_bytes = layers * align(kv_hbm_slots * self.kv_bytes, page)
            weight_hbm = min(weights, hbm_free - kv_hbm_bytes)
            weight_hbm = weight_hbm // page * page
            weight_tier = kv_tier = "mixed"
        else:
            weight_hbm = weights if weight_tier == "hbm" else 0
            available_bytes = capacities[kv_tier] - (reserve if kv_tier == "hbm" else 0)
            if weight_tier == kv_tier:
                available_bytes -= weights
            per_layer = (available_bytes // layers) // page * page
            available = (per_layer // self.kv_bytes - native_page) // native_page * native_page
            requested = available if requested is None else requested
            positive_int(requested, "max_total_tokens")
            if requested > available:
                raise ValueError(f"max_total_tokens exceeds placement capacity {available}")
            pool = requested + native_page
            kv_hbm_slots = pool if kv_tier == "hbm" else 0
        if requested % native_page:
            raise ValueError("max_total_tokens must be native-page aligned")
        self.max_total_tokens, self.pool_tokens = requested, pool
        self.weights = {}
        used = {tier: 0 for tier in capacities}
        weight_tiers = {tier: 0 for tier in capacities}
        for obj in objects:
            allocated = align(obj.bytes, page)
            hbm = min(allocated, weight_hbm) if priority else (allocated if weight_tier == "hbm" else 0)
            extents = []
            if hbm:
                extents.append(Extent(0, min(hbm, obj.bytes), "hbm", used["hbm"]))
                used["hbm"] += hbm
                weight_tiers["hbm"] += hbm
                weight_hbm -= hbm
            if hbm < allocated:
                tier = "hbf" if priority else weight_tier
                extents.append(Extent(hbm, obj.bytes-hbm, tier, used[tier]))
                used[tier] += allocated-hbm
                weight_tiers[tier] += allocated-hbm
            self.weights[obj.id] = tuple(extents)
        self.initial_hbf_pages = used["hbf"] // page
        self.cache_base = used["hbm"]
        used["hbm"] += cache
        self.staging_base = used["hbm"]
        used["hbm"] += staging + workspace
        self.kv = []
        self.kv_arenas = []
        kv_tiers = {tier: 0 for tier in capacities}
        for layer in range(layers):
            extents = []
            portions = [("hbm", 0, kv_hbm_slots)]
            if kv_hbm_slots < pool:
                portions.append(("hbf" if priority else kv_tier, kv_hbm_slots, pool-kv_hbm_slots))
            for tier, first_slot, slots in portions:
                if not slots:
                    continue
                size = slots*self.kv_bytes
                allocated = align(size, page)
                extents.append(Extent(first_slot*self.kv_bytes, size, tier, used[tier]))
                self.kv_arenas.append((tier, used[tier], size, first_slot))
                used[tier] += allocated
                kv_tiers[tier] += allocated
            self.kv.append(tuple(extents))
        self.arenas_by_tier = {tier: [(address, size, first) for home, address, size, first in self.kv_arenas
                                     if home == tier] for tier in capacities}
        self.arena_starts = {tier: [a[0] for a in arenas] for tier, arenas in self.arenas_by_tier.items()}
        used["hbm"] += controller
        for tier, size in used.items():
            if size > capacities[tier]:
                raise ValueError(f"{tier} placement uses {size} bytes, capacity is {capacities[tier]}")
        self.budget = {"capacity_bytes": capacities, "allocated_bytes": used,
            "weight_bytes": model.weight_footprint_bytes, "weight_allocated_bytes_by_tier": weight_tiers,
            "kv_allocated_bytes_by_tier": kv_tiers, "kv_tier": kv_tier,
            "hbm_priority": priority, "hbm_kv_pool_slots": kv_hbm_slots,
            "placement_semantics": "fixed homes; priority reserves full configured KV pool; same slot cutoff in every layer",
            "kv_bytes_per_token_per_layer": self.kv_bytes, "kv_cache_bytes": cache,
            "staging_bytes": staging, "workspace_reserve_bytes": workspace,
            "controller_reserve_bytes": controller, "hbm_application_bytes": capacities["hbm"]-controller,
            "max_total_tokens": requested, "reserved_slots": native_page, "kv_pool_tokens": pool}

    @staticmethod
    def ranges(extents, offset, size):
        end = offset + size
        for extent in extents:
            lo, hi = max(offset, extent.offset), min(end, extent.offset+extent.size)
            if lo < hi:
                yield extent.tier, extent.address+lo-extent.offset, hi-lo

    def page_slots(self, tier, page):
        """Native slot IDs sharing a physical KV page (including partial tokens)."""
        begin, end = page*self.page, (page+1)*self.page
        index = bisect_right(self.arena_starts[tier], begin)-1
        if index >= 0:
            address, size, first_slot = self.arenas_by_tier[tier][index]
            if address < end and begin < address+size:
                lo = max(0, begin-address) // self.kv_bytes
                hi = (min(size, end-address)+self.kv_bytes-1) // self.kv_bytes
                return range(first_slot+lo, first_slot+hi)
        raise ValueError("page is outside KV arenas")
