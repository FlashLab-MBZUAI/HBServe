"""Native-slot placement, finite HBM caching and causal HBFSim execution.

There is no scheduler or KV allocator here. Slot identities and lifetimes come
from SGLang; the only local allocation is physical storage for cache/staging.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, field
import heapq
import json
import math
from pathlib import Path

from hbfsim_client import ResolvedSystemConfig, SimulationSession, Transaction
from hbserve.contracts import RooflineTimingProvider
from .model import NativeCompiler, NativeRequest, dense_model, positive_int


def align(n: int, size: int) -> int:
    return (n + size - 1) // size * size


def merge(intervals):
    result = []
    for begin, end in sorted(intervals):
        if begin >= end:
            continue
        if result and begin <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((begin, end))
    return result


def subtract(intervals, begin, end):
    result = []
    for lo, hi in intervals:
        if hi <= begin or lo >= end:
            result.append((lo, hi))
        else:
            if lo < begin:
                result.append((lo, begin))
            if hi > end:
                result.append((end, hi))
    return result


def missing(intervals, begin, end):
    result = [(begin, end)]
    for lo, hi in intervals:
        result = subtract(result, lo, hi)
    return result


def interval_bytes(intervals):
    return sum(hi - lo for lo, hi in intervals)


@dataclass
class CachePage:
    slot: int
    valid: list[tuple[int, int]] = field(default_factory=list)
    dirty: list[tuple[int, int]] = field(default_factory=list)
    ready: str | None = None


class Graph:
    def __init__(self, prefix):
        self.prefix = prefix
        self.transactions = []

    def emit(self, target="BARRIER", op=None, addr=0, size=0, deps=(), duration=0, stack=None):
        identifier = f"{self.prefix}/{len(self.transactions)}"
        self.transactions.append(Transaction(identifier, target, op, addr, size, 0.0,
            duration, tuple(dict.fromkeys(d for d in deps if d is not None)), stack))
        return identifier


class HardwareSession:
    """One persistent production engine, shared by every native forward."""

    def __init__(self, config: dict, output: Path | None = None, *, initialize=True):
        if not hasattr(SimulationSession, "invalidate_hbf_pages"):
            raise ValueError("native serving requires the current HBFSim client; put the HBFSim checkout before HBServe on PYTHONPATH")
        self.config = config
        self.output = Path(output) if output else None
        self.model = dense_model(config["model"], config["dtype"], config["kv_dtype"])
        self.timing = RooflineTimingProvider(**config["compute"])
        self.weight_tier = config.get("weight_tier", "hbf")
        self.kv_tier = config.get("kv_tier", "hbf")
        if any(t not in {"hbm", "hbf", "external"} for t in (self.weight_tier, self.kv_tier)):
            raise ValueError("weight_tier and kv_tier must be hbm, hbf or external")
        self.architecture = config.get("architecture", "tiered")
        if self.architecture not in {"peer", "tiered"}:
            raise ValueError("architecture must be peer or tiered")
        if self.architecture == "peer" and "external" in (self.weight_tier, self.kv_tier):
            raise ValueError("external backing requires tiered GPU staging; peer is the direct HBF path")
        self.enable_hbf = "hbf" in (self.weight_tier, self.kv_tier)
        self.enable_external = "external" in (self.weight_tier, self.kv_tier)
        self.system = ResolvedSystemConfig.load(tuple(Path(p) for p in config["system_configs"])).resolve(
            Path(config["simulator"]), enable_hbf=self.enable_hbf)
        self.geometry = self.system.hbf_geometry
        self.page = self.geometry.page_size_bytes
        self.chunk = positive_int(config.get("transfer_chunk_bytes", 65536), "transfer_chunk_bytes")
        if self.chunk % self.page:
            raise ValueError("transfer_chunk_bytes must be a multiple of the HBF page size")
        self.background_pages = positive_int(config.get("background_writeback_pages", 2),
                                              "background_writeback_pages", 0)
        self.page_size = positive_int(config.get("page_size", 1), "SGLang page_size")
        cache_bytes = positive_int(config.get("kv_cache_bytes", 0), "kv_cache_bytes", 0)
        if cache_bytes % self.page:
            raise ValueError("kv_cache_bytes must be page aligned")
        if cache_bytes and (self.kv_tier == "hbm" or self.architecture == "peer"):
            raise ValueError("an HBM KV cache requires tiered HBF/external KV placement")
        self.cache_capacity = cache_bytes // self.page
        self.cache: OrderedDict[int, CachePage] = OrderedDict()
        self.free_cache_slots = list(range(self.cache_capacity))
        self.cache_slot_ready = {}
        self.staging_ready = [None, None]
        self.staging_index = 0
        self.backing_ready = {}
        self.finish_times = {}
        self.live_slots = set()
        self.live_pages: dict[int, set[int]] = {}
        self.backed_pages = set()
        self.counters = Counter()
        self.serial = 0
        self.batches = 0
        self.closed = False
        self.last_serving_finish_ns = 0.0
        self.final_drain = None

        capacities = {"hbm": self.system.integer("hbm-capacity-bytes"),
                      "hbf": self.system.logical_hbf_capacity_bytes or 0,
                      "external": int(self.system.external_backing_identity["capacity_bytes"])
                                  if self.enable_external else 0}
        used = {t: 0 for t in capacities}
        self.weights = {}
        for obj in self.model.memory_objects:
            self.weights[obj.id] = used[self.weight_tier]
            used[self.weight_tier] += align(obj.bytes, self.page)
        self.initial_hbf_pages = used["hbf"] // self.page
        self.cache_base = used["hbm"]
        used["hbm"] += cache_bytes
        self.staging_base = used["hbm"]
        staging_bytes = 2 * self.chunk if self.architecture == "tiered" and any(
            t != "hbm" for t in (self.weight_tier, self.kv_tier)) else 0
        used["hbm"] += staging_bytes
        workspace = align(positive_int(config.get("workspace_bytes", 1048576), "workspace_bytes", 0), self.page)
        controller = align(self.system.hbf_ctrl_dram_bytes, self.page) if self.enable_hbf else 0
        used["hbm"] += workspace
        self.kv_base = used[self.kv_tier]
        self.kv_bytes = self.model.layers[0].kv_bytes_per_token
        tier_reserve = controller if self.kv_tier == "hbm" else 0
        per_layer_available = (capacities[self.kv_tier] - tier_reserve - used[self.kv_tier]) // self.model.num_layers
        per_layer_available = per_layer_available // self.page * self.page
        available = (per_layer_available // self.kv_bytes - self.page_size) // self.page_size * self.page_size
        self.max_total_tokens = available if config.get("max_total_tokens") is None else config["max_total_tokens"]
        positive_int(self.max_total_tokens, "max_total_tokens")
        if self.max_total_tokens > available or self.max_total_tokens % self.page_size:
            raise ValueError(f"max_total_tokens must be page aligned and <= placement capacity {available}")
        self.pool_tokens = self.max_total_tokens + self.page_size
        self.layer_stride = align(self.pool_tokens * self.kv_bytes, self.page)
        used[self.kv_tier] += self.model.num_layers * self.layer_stride
        # The engine reserves controller HBM at the top of its address space.
        # Charge its capacity without inserting a second gap before the KV arena.
        used["hbm"] += controller
        for tier in used:
            if used[tier] > capacities[tier]:
                raise ValueError(f"{tier} placement uses {used[tier]} bytes, capacity is {capacities[tier]}")
        self.budget = {"capacity_bytes": capacities, "allocated_bytes": used,
            "weight_bytes": self.model.weight_footprint_bytes, "kv_tier": self.kv_tier,
            "kv_base": self.kv_base, "kv_layer_stride": self.layer_stride,
            "kv_bytes_per_token_per_layer": self.kv_bytes, "kv_cache_bytes": cache_bytes,
            "staging_bytes": staging_bytes, "workspace_reserve_bytes": workspace,
            "controller_reserve_bytes": controller, "hbm_application_bytes": capacities["hbm"]-controller,
            "max_total_tokens": self.max_total_tokens,
            "reserved_slots": self.page_size, "kv_pool_tokens": self.pool_tokens}
        if self.output:
            self.output.mkdir(parents=True, exist_ok=False)
            (self.output / "capacity-budget.json").write_text(json.dumps(self.budget, indent=2))
        if not initialize:
            return  # Capacity planning does not install weights or launch a device.
        self.session = SimulationSession(simulator_path=Path(config["simulator"]), system_config=self.system,
            enable_hbm=True, enable_hbf=self.enable_hbf, enable_external=self.enable_external,
            initial_hbf_logical_pages=self.initial_hbf_pages,
            hbf_wear_output_prefix=self.output / "wear" if self.output and self.enable_hbf else None)

    def _graph(self):
        self.serial += 1
        return Graph(f"native/{self.serial}")

    def _prune(self, now):
        def pending(identifier):
            return identifier if identifier and self.finish_times.get(identifier, math.inf) > now else None
        for entry in self.cache.values():
            entry.ready = pending(entry.ready)
        self.staging_ready = [pending(i) for i in self.staging_ready]
        self.cache_slot_ready = {s: i for s, old in self.cache_slot_ready.items() if (i := pending(old))}
        self.backing_ready = {p: i for p, old in self.backing_ready.items() if (i := pending(old))}
        self.finish_times = {i: t for i, t in self.finish_times.items() if t > now}

    def _retained(self):
        return tuple(sorted({i for i in [*self.staging_ready, *self.cache_slot_ready.values(),
            *self.backing_ready.values(), *(e.ready for e in self.cache.values())] if i}))

    def _submit(self, graph, frontier=None):
        result = self.session.run(graph.transactions, frontier=frontier,
                                  retain=self._retained(), completions=True)
        self.finish_times.update({r.id: r.finish_ns for r in result.completions})
        if self.output:
            receipt = {k: v for k, v in result.receipt.items() if k != "transaction_completions"}
            with (self.output / "batch-receipts.jsonl").open("a") as stream:
                stream.write(json.dumps(receipt) + "\n")
            if not (self.output / "first-dag.json").exists() and len(graph.transactions) > 1:
                (self.output / "first-dag.json").write_text(json.dumps({
                    "transactions": [asdict(t) for t in graph.transactions],
                    "receipt": result.receipt}, indent=2))
        self._prune(result.blocking_finish_ns)
        return result

    def _align_clock(self, now_ns):
        if not math.isfinite(now_ns) or (now_ns < self.last_serving_finish_ns and not
                math.isclose(now_ns, self.last_serving_finish_ns, rel_tol=1e-12, abs_tol=1e-5)):
            raise ValueError("SGLang logical clock moved backwards")
        gap = now_ns - self.session.completed_frontier_ns
        if gap > 1e-6:
            graph = self._graph()
            graph.emit(duration=gap)
            self._submit(graph)
        self._prune(self.session.completed_frontier_ns)

    def _pieces(self, address, size):
        while size:
            page, offset = divmod(address, self.page)
            take = min(size, self.page - offset)
            yield page, offset, offset + take
            size -= take
            address += take

    def _slot_ranges(self, slot):
        for layer in range(self.model.num_layers):
            yield from self._pieces(self.kv_base + layer * self.layer_stride + slot * self.kv_bytes, self.kv_bytes)

    def release(self, slots):
        """Discard dead sectors; invalidate backing only when the entire page dies."""
        dead_pages = set()
        for slot in set(slots):
            if not 0 <= slot < self.pool_tokens:
                raise ValueError("allocator released a slot outside its configured pool")
            if slot not in self.live_slots:
                continue  # Native paged frees also include never-written padding.
            self.live_slots.remove(slot)
            self.counters["released_live_slots"] += 1
            for page, lo, hi in self._slot_ranges(slot):
                entry = self.cache.get(page)
                if entry:
                    before = interval_bytes(entry.dirty)
                    entry.valid = subtract(entry.valid, lo, hi)
                    entry.dirty = subtract(entry.dirty, lo, hi)
                    self.counters["discarded_dirty_bytes"] += before - interval_bytes(entry.dirty)
                self.live_pages[page].discard(slot)
                if not self.live_pages[page]:
                    del self.live_pages[page]
                    dead_pages.add(page)
        for page in dead_pages:
            entry = self.cache.pop(page, None)
            if entry:
                if entry.ready:
                    self.cache_slot_ready[entry.slot] = entry.ready
                heapq.heappush(self.free_cache_slots, entry.slot)
        invalidations = sorted(dead_pages & self.backed_pages)
        if self.kv_tier == "hbf":
            # The current core invalidation is an explicit IO fence. Account its
            # time in the next forward; never silently model it as async TRIM.
            for lo, hi in merge((page, page + 1) for page in invalidations):
                self.serial += 1
                receipt = self.session.invalidate_hbf_pages(f"native/free/{self.serial}",
                    first_lpn=lo, page_count=hi-lo)
                self.counters["hbf_invalidated_pages"] += hi-lo
                if self.output:
                    with (self.output / "lifecycle.jsonl").open("a") as stream:
                        stream.write(json.dumps(receipt) + "\n")
            self._prune(self.session.completed_frontier_ns)
        for page in dead_pages:
            self.backed_pages.discard(page)

    def _observe(self, rows):
        if len({r.rid for r in rows}) != len(rows) or not rows:
            raise ValueError("native batch must have unique nonempty requests")
        for row in rows:
            row.validate(self.pool_tokens, self.model.vocab_size)
            if any(slot < self.page_size for slot in row.slots):
                raise ValueError("native KV snapshot accesses the allocator's reserved page")
            if not set(row.slots[:row.past]) <= self.live_slots:
                raise ValueError("native prefix references KV that was never written or already freed")
        for row in rows:
            for slot in row.slots[row.past:]:
                if slot in self.live_slots:
                    continue
                self.live_slots.add(slot)
                for page, _, _ in self._slot_ranges(slot):
                    self.live_pages.setdefault(page, set()).add(slot)
        self.counters["peak_live_slots"] = max(self.counters["peak_live_slots"], len(self.live_slots))

    def _link(self, graph, tier, address, size, write, deps):
        if tier != "hbf":
            return graph.emit(deps=deps)
        counts = Counter()
        for page, lo, hi in self._pieces(address, size):
            counts[self.geometry.stack_for_logical_page(page)] += hi-lo
        terminals = [graph.emit("D2D_HBM_TO_HBF" if write else "D2D_HBF_TO_HBM",
            "W" if write else "R", 0, count, deps, stack=stack) for stack, count in counts.items()]
        return graph.emit(deps=terminals)

    def _backing(self, graph, tier, op, address, size, deps, *, kv=False):
        pages = {p for p, _, _ in self._pieces(address, size)}
        waits = [self.backing_ready.get((tier, p)) for p in pages]
        terminal = graph.emit({"hbm": "HBM", "hbf": "HBF_LOGICAL", "external": "EXTERNAL"}[tier],
                              op, address, size, (*deps, *waits))
        if op == "W":
            for p in pages:
                self.backing_ready[tier, p] = terminal
            if kv:
                self.backed_pages.update(pages)
        return terminal

    def _transfer(self, graph, tier, op, address, size, hbm_address, deps, *, kv=False):
        if op == "R":
            read = self._backing(graph, tier, "R", address, size, deps, kv=kv)
            link = self._link(graph, tier, address, size, False, (read,))
            return graph.emit("HBM", "W", hbm_address, size, (link,))
        read = graph.emit("HBM", "R", hbm_address, size, deps)
        link = self._link(graph, tier, address, size, True, (read,))
        return self._backing(graph, tier, "W", address, size, (link,), kv=kv)

    def _uncached(self, graph, tier, op, address, size, deps, *, kv=False):
        if tier == "hbm" or self.architecture == "peer":
            return self._backing(graph, tier, op, address, size, deps, kv=kv)
        terminals = []
        for offset in range(0, size, self.chunk):
            take = min(self.chunk, size-offset)
            slot = self.staging_index % 2
            self.staging_index += 1
            stage = self.staging_base + slot*self.chunk
            waits = (*deps, self.staging_ready[slot])
            if op == "W":
                produced = graph.emit("HBM", "W", stage, take, waits)
                done = self._transfer(graph, tier, op, address+offset, take, stage, (produced,), kv=kv)
            else:
                loaded = self._transfer(graph, tier, op, address+offset, take, stage, waits, kv=kv)
                done = graph.emit("HBM", "R", stage, take, (loaded,))
            self.staging_ready[slot] = done
            terminals.append(done)
        return graph.emit(deps=terminals)

    def _writeback(self, graph, page, entry):
        for lo, hi in entry.dirty:
            entry.ready = self._transfer(graph, self.kv_tier, "W", page*self.page+lo, hi-lo,
                self.cache_base+entry.slot*self.page+lo, (entry.ready,), kv=True)
            self.counters["writeback_bytes"] += hi-lo
        entry.dirty = []
        if entry.ready:
            self.cache_slot_ready[entry.slot] = entry.ready
        return entry.ready

    def _cache_page(self, graph, page):
        if page in self.cache:
            self.cache.move_to_end(page)
            return self.cache[page]
        if self.free_cache_slots:
            slot = heapq.heappop(self.free_cache_slots)
        else:
            old_page, old = self.cache.popitem(last=False)
            self._writeback(graph, old_page, old)
            slot = old.slot
            if old.ready:
                self.cache_slot_ready[slot] = old.ready
            self.counters["cache_evictions"] += 1
        entry = CachePage(slot=slot, ready=self.cache_slot_ready.get(slot))
        self.cache[page] = entry
        return entry

    def _cached(self, graph, op, address, size, deps):
        terminals = []
        for page, lo, hi in self._pieces(address, size):
            entry = self._cache_page(graph, page)
            hbm = self.cache_base+entry.slot*self.page
            waits = (*deps, entry.ready)
            if op == "R":
                gaps = missing(entry.valid, lo, hi)
                self.counters["cache_hit_bytes"] += hi-lo-interval_bytes(gaps)
                self.counters["cache_miss_bytes"] += interval_bytes(gaps)
                for begin, end in gaps:
                    entry.ready = self._transfer(graph, self.kv_tier, "R", page*self.page+begin,
                        end-begin, hbm+begin, waits, kv=True)
                    waits = (entry.ready,)
                entry.ready = graph.emit("HBM", "R", hbm+lo, hi-lo, waits)
            else:
                entry.ready = graph.emit("HBM", "W", hbm+lo, hi-lo, waits)
                entry.dirty = merge([*entry.dirty, (lo, hi)])
            entry.valid = merge([*entry.valid, (lo, hi)])
            self.cache_slot_ready[entry.slot] = entry.ready
            terminals.append(entry.ready)
        return graph.emit(deps=terminals)

    def _kv_ranges(self, row, layer, offset, size):
        start, within = divmod(offset, self.kv_bytes)
        if within or size % self.kv_bytes:
            raise ValueError("semantic KV access is not token aligned")
        slots = row.slots[start:start+size//self.kv_bytes]
        if len(slots)*self.kv_bytes != size:
            raise ValueError("semantic KV access exceeds the native slot map")
        # Preserve actual order and coalesce only truly adjacent native slots.
        runs = []
        for slot in slots:
            addr = self.kv_base+layer*self.layer_stride+slot*self.kv_bytes
            if runs and runs[-1][0]+runs[-1][1] == addr:
                runs[-1] = (runs[-1][0], runs[-1][1]+self.kv_bytes)
            else:
                runs.append((addr, self.kv_bytes))
        return runs

    def run_batch(self, rows: list[NativeRequest], *, now_ns: float, freed_slots=()):
        if self.closed:
            raise RuntimeError("native benchmark is already finalized")
        self._align_clock(now_ns)
        self.release(freed_slots)
        self._observe(rows)
        canonical = NativeCompiler(self.model, rows, self.timing).batch(self.batches, now_ns)
        by_key = {r.key: r for r in rows}
        graph = self._graph()
        projection = {}
        before = self.counters.copy()
        for operation in canonical.operations:
            deps = tuple(projection[d] for d in operation.dependencies)
            if operation.op is None:
                terminal = graph.emit(deps=deps, duration=operation.duration_ns)
            elif operation.object_id in self.weights:
                terminal = self._uncached(graph, self.weight_tier, operation.op,
                    self.weights[operation.object_id]+operation.offset, operation.bytes, deps)
                self.counters["weight_read_bytes"] += operation.bytes
            else:
                parts = operation.object_id.split("/")
                row, layer = by_key[parts[1]], int(parts[5])
                terminals = []
                for address, size in self._kv_ranges(row, layer, operation.offset, operation.bytes):
                    terminals.append(self._cached(graph, operation.op, address, size, deps)
                        if self.cache_capacity else self._uncached(graph, self.kv_tier,
                            operation.op, address, size, deps, kv=True))
                terminal = graph.emit(deps=terminals)
                self.counters["kv_read_bytes" if operation.op == "R" else "kv_write_bytes"] += operation.bytes
            projection[operation.id] = terminal
        foreground = projection[canonical.operations[-1].id]
        flushed = 0
        for page, entry in self.cache.items():
            if flushed >= self.background_pages:
                break
            if entry.dirty:
                self._writeback(graph, page, entry)
                flushed += 1
        result = self._submit(graph, (foreground,))
        self.last_serving_finish_ns = result.blocking_finish_ns
        record = {"batch_id": self.batches, "start_ns": now_ns,
            "forward_finish_ns": result.blocking_finish_ns,
            "issued_finish_ns": result.finish_ns,
            "latency_ns": result.blocking_finish_ns-now_ns,
            "kind": canonical.schedule.kind, "requests": [asdict(r) for r in rows],
            "freed_slots": list(freed_slots), "live_slots": len(self.live_slots),
            "dirty_cache_bytes": sum(interval_bytes(e.dirty) for e in self.cache.values()),
            "background_pages": flushed, "canonical_sha256": canonical.digest,
            "transaction_sha256": result.receipt["transaction_trace_sha256"],
            "traffic": dict(self.counters-before)}
        self.batches += 1
        if self.output:
            with (self.output / "native_batches.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        return record

    def finalize(self, freed_slots=()):
        if self.closed:
            return
        begin = self.last_serving_finish_ns
        self.release(freed_slots)
        graph = self._graph()
        for page, entry in self.cache.items():
            if entry.dirty:
                self._writeback(graph, page, entry)
        if graph.transactions:
            self._submit(graph)
        self.session.close()
        self.closed = True
        receipt = self.session.source_receipt()
        self.final_drain = {"serving_finish_ns": begin,
            "frontend_finish_ns": self.session.completed_frontier_ns,
            "excluded_from_request_latency": True,
            "physical_stop": receipt.get("final_measurement")}
        if self.output:
            (self.output / "result.json").write_text(json.dumps(self.report(), indent=2))

    def report(self):
        return {"schema": "hbserve.sglang_hbfsim.v1", "batches": self.batches,
            "model": self.model.canonical(), "budget": self.budget,
            "traffic": dict(self.counters), "live_slots": len(self.live_slots),
            "dirty_cache_bytes": sum(interval_bytes(e.dirty) for e in self.cache.values()),
            "finalized": self.closed, "drain": self.final_drain,
            "engine": self.session.source_receipt(),
            "scope": {"scheduler": "native_sglang", "kv_allocator": "native_sglang",
                "memory_engine": "production_hbfsim", "compute": self.timing.canonical(),
                "traffic": "HBServe object-level DAG; existing KV once per request per layer",
                "hbf_invalidation": "whole-page, after issued-IO fence",
                "external_free": "logical liveness only; no media TRIM",
                "limits": ["Single worker, dense full-context attention; no MoE, HiCache or collectives.",
                    "GPU kernels and numerical tokens are not executed; compute is uncalibrated roofline.",
                    "Scratch traffic and kernel cache effects require an independent kernel trace."]}}

    def close(self):
        """Close the child even when a native forward fails."""
        if not self.closed:
            self.session.close()
            self.closed = True
