#!/usr/bin/env python3
"""HBServe's closed-loop request scheduler (token-budgeted mixed iterations)."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Protocol, Sequence

from hbserve.compiler import HBServeCompiler
from hbserve.contracts import (
    BatchSlice,
    CanonicalServingBatch,
    ModelSpec,
    RequestSpec,
    RequestTrace,
    ScheduledBatch,
    SchedulerPolicy,
    HBServeError,
    canonical_sha256,
)


RUN_SCHEMA = {"name": "hbserve.run", "version": 2}
PREEMPTION_POLICY = "youngest_request_first_swap_or_recompute_v1"


def _nearest_rank_percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    result: dict[str, float] = {}
    for label, percentile in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99)):
        index = max(0, math.ceil(percentile * len(ordered)) - 1)
        result[label] = ordered[index]
    return result


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


@dataclass(frozen=True)
class BatchExecution:
    batch_id: int
    batch_origin_ns: float
    finish_ns: float
    canonical_sha256: str
    mapped_sha256: str
    remap_receipt: Mapping[str, Any]
    physical_completion: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.batch_id < 0:
            raise HBServeError("batch execution ID is negative")
        for name, value in (
            ("batch_origin_ns", self.batch_origin_ns),
            ("finish_ns", self.finish_ns),
        ):
            if not math.isfinite(value) or value < 0:
                raise HBServeError(f"batch execution {name} is invalid")
        if self.finish_ns < self.batch_origin_ns:
            raise HBServeError("batch execution finishes before its origin")
        for name, value in (
            ("canonical", self.canonical_sha256),
            ("mapped", self.mapped_sha256),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise HBServeError(
                    f"batch execution {name} digest is malformed"
                )


class HBServeExecutor(Protocol):
    """Physical execution plane: KV capacity, migration, and batch completion."""

    @property
    def frontier_ns(self) -> float: ...

    @property
    def execution_identity(self) -> Mapping[str, Any]: ...

    def admit_request(self, request: RequestSpec) -> int: ...

    def can_reserve(self, slices: Sequence[BatchSlice]) -> bool: ...

    def reserve(
        self, batch_id: int, slices: Sequence[BatchSlice]
    ) -> Mapping[str, Any]: ...

    def preempt_request(self, request_id: str) -> Mapping[str, Any]: ...

    def release_request(self, request_id: str) -> None: ...

    def execute(self, batch: CanonicalServingBatch) -> BatchExecution: ...


@dataclass
class _RequestState:
    request: RequestSpec
    order: int
    admitted: bool = False
    processed: int = 0
    emitted: int = 0
    prefill_target: int = 0
    hot_kv_blocks: bool = False
    first_token_ns: float | None = None
    completion_ns: float | None = None
    preemptions: int = 0
    prefix_hit_tokens: int = 0

    def __post_init__(self) -> None:
        self.prefill_target = self.request.prompt_tokens

    @property
    def complete(self) -> bool:
        return self.completion_ns is not None

    @property
    def phase(self) -> str:
        return "prefill" if self.processed < self.prefill_target else "decode"

    @property
    def sort_key(self) -> tuple[float, int]:
        return (self.request.arrival_ns, self.order)


class HBServeEngine:
    """vLLM-V1-style continuous batching with physical feedback.

    Every iteration admits all runnable decode requests plus prefill chunks of
    waiting requests within the token and request budgets, so an arrival is
    prefilled in the next iteration rather than after the in-flight cohort
    drains.  KV blocks are reserved before the iteration is compiled; when the
    HBM KV pool cannot hold the iteration even after migrating every waiting
    request to the cold tier, the youngest request in the iteration is
    preempted (swapped out, or freed for recompute without a cold tier).  One
    batch is submitted only after the previous mapped batch returns its
    physical completion, so HBF queueing and migration time change later
    admission, batching, and completion times rather than being added as an
    offline correction.
    """

    def __init__(
        self,
        *,
        models: Mapping[str, ModelSpec],
        request_trace: RequestTrace,
        compiler: HBServeCompiler,
        executor: HBServeExecutor,
        policy: SchedulerPolicy,
        request_groups: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self.models = dict(models)
        self.request_trace = request_trace
        self.compiler = compiler
        self.executor = executor
        self.policy = policy
        if self.compiler.request_trace.digest != request_trace.digest:
            raise HBServeError(
                "serving engine/compiler request traces differ"
            )
        if set(self.compiler.models) != set(self.models):
            raise HBServeError(
                "serving engine/compiler model catalogs differ"
            )
        self._states = [
            _RequestState(request=request, order=index)
            for index, request in enumerate(request_trace.requests)
        ]
        self._by_id = {state.request.request_id: state for state in self._states}
        self._batch_records: list[dict[str, Any]] = []
        self._preemptions: list[dict[str, Any]] = []
        # V1 schedules existing running requests in admission order, including
        # unfinished prefill chunks, before admitting waiting requests.
        self._running: list[_RequestState] = []
        self._resumed: list[_RequestState] = []
        self._request_groups = [dict(id=g['id'],request_ids=list(g['request_ids'])) for g in (request_groups or ())]
        if self._request_groups:
            identifiers = [r for g in self._request_groups for r in g['request_ids']]
            if len(identifiers)!=len(set(identifiers)) or set(identifiers)!=set(self._by_id):
                raise HBServeError('request groups must partition the offered requests')
        self._group_index = 0
        self._group_releases: dict[str, float] = {}

    def _runnable_states(self):
        if not self._request_groups:
            return [s for s in self._states if not s.complete]
        while self._group_index < len(self._request_groups):
            group = self._request_groups[self._group_index]
            if group['id'] not in self._group_releases:
                release=self.executor.frontier_ns
                self._group_releases[group['id']]=release
                for identifier in group['request_ids']:
                    state=self._by_id[identifier]
                    state.request=replace(state.request,arrival_ns=release+state.request.arrival_ns)
            states = [self._by_id[r] for r in group['request_ids'] if not self._by_id[r].complete]
            if states:
                return states
            self._group_index += 1
        return []

    def _slice_for(self, state: _RequestState, budget_tokens: int) -> BatchSlice:
        request = state.request
        if state.phase == "decode":
            if state.processed != request.prompt_tokens + state.emitted - 1:
                raise HBServeError(
                    f"request {request.request_id} decode frontier is inconsistent"
                )
            return BatchSlice(
                request_id=request.request_id,
                token_begin=state.processed,
                token_count=1,
                context_tokens_before=state.processed,
                emits_output=True,
                phase="decode",
            )
        remaining = state.prefill_target - state.processed
        count = min(remaining, self.policy.prefill_chunk_tokens, budget_tokens)
        end = state.processed + count
        return BatchSlice(
            request_id=request.request_id,
            token_begin=state.processed,
            token_count=count,
            context_tokens_before=state.processed,
            emits_output=(
                end == state.prefill_target and end >= request.prompt_tokens
            ),
            phase="prefill",
        )

    def _preempt(self, batch_id: int, state: _RequestState) -> None:
        receipt = dict(self.executor.preempt_request(state.request.request_id))
        mode = receipt.get("mode")
        processed_before = state.processed
        if mode == "recompute":
            # The discarded context (prompt or prompt + generated so far)
            # is re-fed as one prefill; its last chunk produces the next
            # token, so an emitted first token keeps its timestamp.
            state.processed = 0
            state.prefill_target = (
                state.request.prompt_tokens
                if processed_before < state.request.prompt_tokens
                else processed_before + 1
            )
        elif mode != "swap_out":
            raise HBServeError(f"executor returned unknown preemption mode {mode!r}")
        state.hot_kv_blocks = False
        if state in self._running:
            self._running.remove(state)
        if state in self._resumed:
            self._resumed.remove(state)
        self._resumed.insert(0, state)
        state.preemptions += 1
        self._preemptions.append(
            {
                "batch_id": batch_id,
                "request_id": state.request.request_id,
                "mode": mode,
                "processed_tokens_before": processed_before,
                "blocks": int(receipt.get("blocks", 0)),
                "bytes": int(receipt.get("bytes", 0)),
            }
        )

    def _next_schedule(
        self, batch_id: int
    ) -> tuple[ScheduledBatch, Mapping[str, Any]]:
        frontier = self.executor.frontier_ns
        unfinished = self._runnable_states()
        if not unfinished:
            raise HBServeError("scheduler was called after completion")
        scheduler_now = frontier
        if not any(state.request.arrival_ns <= frontier for state in unfinished):
            scheduler_now = min(state.request.arrival_ns for state in unfinished)
        arrived = sorted(
            (
                state
                for state in unfinished
                if state.request.arrival_ns <= scheduler_now
            ),
            key=lambda state: state.sort_key,
        )
        running = [state for state in self._running if state in arrived]
        waiting = [state for state in self._resumed if state in arrived]
        waiting += [state for state in arrived if state not in running and state not in waiting]
        # One model per iteration: the oldest running request's model, or the
        # oldest arrival's when nothing is decoding.
        model_id = (running or waiting)[0].request.model_id
        budget_tokens = self.policy.max_batch_tokens
        selected: list[tuple[_RequestState, BatchSlice]] = []
        reservations = []
        preempted = False
        for state in running:
            if state not in self._running:
                continue
            if state.request.model_id != model_id:
                continue
            if budget_tokens == 0:
                break
            batch_slice = self._slice_for(state, budget_tokens)
            while not self.executor.can_reserve(tuple(s for _, s in selected) + (batch_slice,)):
                victim = self._running[-1]
                for index, (chosen, item) in enumerate(selected):
                    if chosen is victim:
                        budget_tokens += item.token_count
                        selected.pop(index)
                        break
                self._preempt(batch_id, victim)
                preempted = True
                if victim is state:
                    break
            if state in self._running:
                selected.append((state, batch_slice))
                budget_tokens -= batch_slice.token_count

        # Running-request preemption is finished before publishing planned
        # full prefix blocks. Later admissions can share blocks produced in
        # this same synchronous iteration, as in vLLM's allocation path.
        if selected:
            reservations.append(self.executor.reserve(batch_id, tuple(s for _,s in selected)))
        # A new admission may wait for capacity; it must not evict an already
        # runnable request merely to fill this iteration's token budget.
        for state in (() if preempted else waiting):
            if state.request.model_id != model_id:
                continue
            if budget_tokens == 0 or len(self._running) >= self.policy.max_batch_requests:
                break
            newly_admitted = not state.admitted
            if not state.admitted:
                cached_tokens = self.executor.admit_request(state.request)
                if isinstance(cached_tokens, bool) or not isinstance(cached_tokens, int) or not 0 <= cached_tokens < state.request.prompt_tokens:
                    raise HBServeError("executor returned an invalid cached prefix length")
                state.processed = cached_tokens
                state.prefix_hit_tokens = cached_tokens
                state.hot_kv_blocks = cached_tokens > 0
                state.admitted = True
            batch_slice = self._slice_for(state, budget_tokens)
            if not self.executor.can_reserve(tuple(s for _, s in selected) + (batch_slice,)):
                if newly_admitted:
                    self.executor.release_request(state.request.request_id)
                    state.admitted = False
                    state.processed = state.prefix_hit_tokens = 0
                    state.hot_kv_blocks = False
                break
            selected.append((state, batch_slice))
            reservations.append(self.executor.reserve(batch_id, tuple(s for _,s in selected)))
            self._running.append(state)
            if state in self._resumed:
                self._resumed.remove(state)
            budget_tokens -= batch_slice.token_count
        slices = tuple(item for _, item in selected)
        if not slices:
            if preempted:
                return self._next_schedule(batch_id)
            raise HBServeError("KV pool cannot hold one schedulable chunk of the oldest waiting request")
        receipt = dict(reservations[-1])
        for key in ('allocated_blocks','swap_out_blocks','swap_in_blocks','prefix_cache_evictions'):
            receipt[key] = sum(r.get(key,0) for r in reservations)
        for key in ('swap_out_requests','swap_in_requests'):
            receipt[key] = list(dict.fromkeys(x for r in reservations for x in r.get(key,())))
        for state, _ in selected:
            state.hot_kv_blocks = True
        return (
            ScheduledBatch(
                batch_id=batch_id,
                model_id=model_id,
                slices=slices,
                not_before_ns=scheduler_now,
            ),
            receipt,
        )

    def _apply_completion(
        self, schedule: ScheduledBatch, finish_ns: float
    ) -> None:
        for batch_slice in schedule.slices:
            state = self._by_id[batch_slice.request_id]
            if state.processed != batch_slice.token_begin:
                raise HBServeError(
                    "completion does not advance the exact token frontier"
                )
            state.processed = batch_slice.token_end
            if batch_slice.emits_output:
                state.emitted += 1
                if state.emitted == 1:
                    state.first_token_ns = finish_ns
                if state.emitted > state.request.output_tokens:
                    raise HBServeError(
                        "request emitted more tokens than requested"
                    )
                if state.emitted == state.request.output_tokens:
                    if state.processed != state.request.processed_input_tokens:
                        raise HBServeError(
                            "request completed without processing its lifetime"
                        )
                    state.completion_ns = finish_ns
                    self.executor.release_request(state.request.request_id)
                    state.hot_kv_blocks = False
                    if state in self._running:
                        self._running.remove(state)

    def run(self) -> dict[str, Any]:
        batch_id = 0
        while any(not state.complete for state in self._states):
            schedule, reservation = self._next_schedule(batch_id)
            canonical = self.compiler.compile(schedule)
            execution = self.executor.execute(canonical)
            if execution.batch_id != batch_id:
                raise HBServeError(
                    "executor returned a different batch identity"
                )
            if execution.canonical_sha256 != canonical.digest:
                raise HBServeError(
                    "executor completion changed the canonical workload"
                )
            if execution.finish_ns + 1e-9 < schedule.not_before_ns:
                raise HBServeError(
                    "executor completed a batch before its scheduling release"
                )
            if not math.isclose(
                self.executor.frontier_ns,
                execution.finish_ns,
                rel_tol=1e-12,
                abs_tol=1e-6,
            ):
                raise HBServeError(
                    "executor frontier differs from the returned completion"
                )
            self._apply_completion(schedule, execution.finish_ns)
            self._batch_records.append(
                {
                    "batch_id": batch_id,
                    "kind": schedule.kind,
                    "schedule": schedule.canonical(),
                    "kv_reservation": dict(reservation),
                    "canonical_sha256": canonical.digest,
                    "logical_bytes": canonical.logical_bytes,
                    "canonical_workload": canonical.audit_summary(),
                    "timing_model": canonical.timing_model,
                    "timing_evidence_state": canonical.timing_evidence_state,
                    "batch_origin_ns": execution.batch_origin_ns,
                    "finish_ns": execution.finish_ns,
                    "mapped_sha256": execution.mapped_sha256,
                    "remap": dict(execution.remap_receipt),
                    "physical_completion": dict(execution.physical_completion),
                }
            )
            # Release this iteration's graph before constructing the next one.
            # The batch record already owns its complete audit summaries.
            del canonical, execution
            batch_id += 1
        return self._result()

    def _result(self) -> dict[str, Any]:
        includes_compute = self.compiler.timing.includes_compute
        timing_model = self.compiler.timing.timing_model
        first_key = (
            "ttft_ns" if includes_compute else "memory_critical_path_first_token_ns"
        )
        per_token_key = (
            "tpot_ns" if includes_compute else "memory_critical_path_per_token_ns"
        )
        request_rows: list[dict[str, Any]] = []
        for state in self._states:
            if (
                state.first_token_ns is None
                or state.completion_ns is None
                or state.emitted != state.request.output_tokens
            ):
                raise HBServeError(
                    f"request {state.request.request_id} did not complete exactly"
                )
            output_intervals = state.request.output_tokens - 1
            request_rows.append(
                {
                    "request_id": state.request.request_id,
                    "model_id": state.request.model_id,
                    "arrival_ns": state.request.arrival_ns,
                    "prompt_tokens": state.request.prompt_tokens,
                    "output_tokens": state.request.output_tokens,
                    "first_token_ns": state.first_token_ns,
                    "completion_ns": state.completion_ns,
                    "preemptions": state.preemptions,
                    "prefix_hit_tokens": state.prefix_hit_tokens,
                    first_key: state.first_token_ns - state.request.arrival_ns,
                    "e2e_ns": state.completion_ns - state.request.arrival_ns,
                    per_token_key: (
                        None
                        if output_intervals == 0
                        else (
                            (state.completion_ns - state.first_token_ns)
                            / output_intervals
                        )
                    ),
                }
            )
        first_arrival = min(row["arrival_ns"] for row in request_rows)
        last_completion = max(row["completion_ns"] for row in request_rows)
        duration_ns = last_completion - first_arrival
        total_output_tokens = sum(row["output_tokens"] for row in request_rows)
        timing_states = {
            row["timing_evidence_state"] for row in self._batch_records
        }
        execution_identity = dict(self.executor.execution_identity)
        role_totals: dict[str, dict[str, int]] = {}
        token_id_sources: dict[str, int] = {}
        moe_routes: dict[tuple[str, int, int], int] = {}
        kinds = {"prefill": 0, "decode": 0, "mixed": 0}
        swap_out_bytes = 0
        swap_in_bytes = 0
        for batch in self._batch_records:
            kinds[batch["kind"]] += 1
            workload = batch["canonical_workload"]
            for role, row in workload["roles"].items():
                total = role_totals.setdefault(
                    role,
                    {
                        "operations": 0,
                        "read_bytes": 0,
                        "write_bytes": 0,
                        "barriers": 0,
                    },
                )
                for field in total:
                    total[field] += int(row[field])
            for source, count in workload["token_id_sources"].items():
                token_id_sources[source] = (
                    token_id_sources.get(source, 0) + int(count)
                )
            model_id = str(batch["schedule"]["model_id"])
            for row in workload["moe_expert_routes"]:
                key = (model_id, int(row["layer"]), int(row["expert"]))
                moe_routes[key] = moe_routes.get(key, 0) + int(
                    row["routed_tokens"]
                )
            overhead = batch["remap"].get("policy_overhead", {})
            swap_out_bytes += int(overhead.get("kv_swap_out_bytes", 0))
            swap_in_bytes += int(overhead.get("kv_swap_in_bytes", 0))
        physical_memory_timing = (
            execution_identity.get("physical_memory_timing") is True
        )
        absolute_memory_timing = (
            execution_identity.get(
                "absolute_memory_timing_claim_eligible"
            )
            is True
        )
        backend_ttft_tpot = (
            execution_identity.get("ttft_tpot_claim_eligible") is True
        )
        model_sources_qualified = all(
            model.provenance["kind"]
            in {"checkpoint_manifest", "published_descriptor"}
            for model in self.models.values()
        )
        request_source_qualified = self.request_trace.provenance.kind in {
            "production_trace",
            "model_generated_trace",
        }
        uses_moe = any(
            layer.is_moe
            for model in self.models.values()
            for layer in model.layers
        )
        router_provenance = (
            None
            if self.compiler.router is None
            else self.compiler.router.provenance.canonical()
        )
        router_source_qualified = (
            None
            if not uses_moe
            else (
                self.compiler.router is not None
                and self.compiler.router.provenance.kind
                in {"production_trace", "model_generated_trace"}
            )
        )
        first_values = [float(row[first_key]) for row in request_rows]
        per_token_values = [
            float(row[per_token_key])
            for row in request_rows
            if row[per_token_key] is not None
        ]
        summary: dict[str, Any] = {
            "timing_model": timing_model,
            "includes_compute": includes_compute,
            "requests": len(request_rows),
            "output_tokens": total_output_tokens,
            "prefix_hit_tokens": sum(state.prefix_hit_tokens for state in self._states),
            "prefix_hit_requests": sum(state.prefix_hit_tokens > 0 for state in self._states),
            "iterations": len(self._batch_records),
            "preemptions": len(self._preemptions),
            "first_arrival_ns": first_arrival,
            "last_completion_ns": last_completion,
            "duration_ns": duration_ns,
            "mean_e2e_ns": _mean([float(row["e2e_ns"]) for row in request_rows]),
            "latency_percentiles": {
                "method": "nearest_rank_v1",
                first_key: _nearest_rank_percentiles(first_values),
                "e2e_ns": _nearest_rank_percentiles(
                    [float(row["e2e_ns"]) for row in request_rows]
                ),
                per_token_key: _nearest_rank_percentiles(per_token_values),
            },
        }
        if includes_compute:
            summary["mean_ttft_ns"] = _mean(first_values)
            summary["mean_tpot_ns"] = _mean(per_token_values)
            summary["request_throughput_per_second"] = (
                None if duration_ns == 0 else len(request_rows) * 1e9 / duration_ns
            )
            summary["output_token_throughput_per_second"] = (
                None
                if duration_ns == 0
                else total_output_tokens * 1e9 / duration_ns
            )
        else:
            summary["mean_memory_critical_path_first_token_ns"] = _mean(
                first_values
            )
            summary["mean_memory_critical_path_per_token_ns"] = _mean(
                per_token_values
            )
            summary["ttft_tpot_omitted_because"] = (
                "memory_only timing carries no compute; the figures above are "
                "memory critical paths, not TTFT/TPOT"
            )
        result = {
            "schema": RUN_SCHEMA,
            "result": "pass",
            "request_trace": {
                "sha256": self.request_trace.digest,
                "provenance": self.request_trace.provenance.canonical(),
            },
            "models": {
                model_id: {
                    "sha256": model.digest,
                    "weight_footprint_bytes": model.weight_footprint_bytes,
                    "kv_bytes_per_token": model.kv_bytes_per_token,
                    "layers": model.num_layers,
                    "provenance": dict(model.provenance),
                }
                for model_id, model in sorted(self.models.items())
            },
            "scheduler": {
                "policy": self.policy.canonical(),
                "kv_allocation_policy": execution_identity.get("placement", {})
                .get("kv", {})
                .get("allocation_policy"),
                "preemption_policy": PREEMPTION_POLICY,
                "closed_loop_memory_feedback": True,
                "batches": len(self._batch_records),
                "iterations_by_kind": kinds,
                "preemptions": list(self._preemptions),
                "kv_migration_bytes": {
                    "swap_out": swap_out_bytes,
                    "swap_in": swap_in_bytes,
                },
            },
            "timing": {
                "model": timing_model,
                "includes_compute": includes_compute,
                "provider": self.compiler.timing.canonical(),
                "prefetch_depth": self.compiler.prefetch_depth,
                "evidence_states": sorted(timing_states),
            },
            "execution": execution_identity,
            "input_evidence": {
                "model_sources_sha_bound": model_sources_qualified,
                "request_source_sha_bound": request_source_qualified,
                "router_source_sha_bound": router_source_qualified,
                "router": {
                    "sha256": self.compiler.router.digest,
                    "provenance": router_provenance,
                }
                if self.compiler.router is not None
                else None,
            },
            "eligibility": {
                "derived_memory_workload": True,
                "measured_gpu_memory_trace": False,
                "capacity_and_byte_accounting": True,
                "modeled_physical_memory_service": physical_memory_timing,
                "absolute_memory_latency_claim": absolute_memory_timing,
                "address_locality_or_tile_claim": False,
                "source_qualified_model_footprints": model_sources_qualified,
                "source_qualified_request_trace": request_source_qualified,
                "source_qualified_router_trace": router_source_qualified,
                "modeled_compute_included": includes_compute,
                "ttft_tpot_reported": includes_compute,
                "ttft_tpot_slo_claim": (
                    timing_states == {"calibrated"}
                    and absolute_memory_timing
                    and backend_ttft_tpot
                ),
                "timing_evidence_states": sorted(timing_states),
                "claim_boundary": (
                    "Measured A100 operator backend with paged/ragged demand; "
                    "request-scheduler and complete-backend validation are separate evidence. "
                    "Native HBF timing and arbitrary production workloads are not hardware-validated."
                    if timing_model == 'gpu_calibrated' else
                    "object-level derived memory demand with roofline, sensitivity or memory-only "
                    "timing; no end-to-end serving calibration"
                ),
            },
            "workload_accounting": {
                "logical_bytes": sum(
                    int(batch["logical_bytes"])
                    for batch in self._batch_records
                ),
                "roles": {
                    role: role_totals[role] for role in sorted(role_totals)
                },
                "token_id_sources": dict(sorted(token_id_sources.items())),
                "moe_expert_routes": [
                    {
                        "model_id": model_id,
                        "layer": layer,
                        "expert": expert,
                        "routed_tokens": count,
                    }
                    for (model_id, layer, expert), count in sorted(
                        moe_routes.items()
                    )
                ],
            },
            "summary": summary,
            "requests": request_rows,
            "batches": self._batch_records,
        }
        if self._request_groups:
            by_id = {r['request_id']:r for r in request_rows}
            groups = []
            for group in self._request_groups:
                selected = [by_id[r] for r in group['request_ids']]
                release = max(self._group_releases[group['id']], min(r['arrival_ns'] for r in selected))
                finish = max(r['completion_ns'] for r in selected)
                ids = set(group['request_ids'])
                groups.append(dict(group,arrival_ns=release,completion_ns=finish,group_end_to_end_ns=finish-release,
                    requests=len(selected),output_tokens=sum(r['output_tokens'] for r in selected),
                    batch_ids=[b['schedule']['batch_id'] for b in self._batch_records
                        if any(s['request_id'] in ids for s in b['schedule']['slices'])]))
            result['request_groups'] = groups
            result['scheduler']['request_group_release'] = 'previous_group_actual_completion'
        result["run_sha256"] = canonical_sha256(result)
        return result
