#!/usr/bin/env python3
"""Compile scheduled LLM iterations into placement-independent object accesses."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping

from hbserve.contracts import (
    BatchSlice,
    CanonicalServingBatch,
    MemoryObject,
    MemoryOnlyTimingProvider,
    ModelSpec,
    RequestSpec,
    RequestTrace,
    RouterProvider,
    ScheduledBatch,
    SemanticOperation,
    HBServeError,
    TimingProvider,
)


DEFAULT_PREFETCH_DEPTH = 1


class HBServeCompiler:
    """Translate one scheduler iteration into a causal object-access DAG.

    The compiler knows model/request semantics but never chooses a memory tier
    or physical address.  Its traffic model is deliberately IO-optimal at the
    object level: one shared weight object per iteration, one selected expert
    object per unique expert in an iteration/layer, one read of each request's
    existing KV context, and one write of every newly processed token's KV.

    Per layer ``L`` the DAG is:

    * ``memory(L)``: weight reads and KV context reads.  They may start once
      ``compute(L - 1 - prefetch_depth)`` has finished, i.e. the layer buffer
      that ``L`` reuses is free; with the default depth of one layer, the
      weights of ``L + 1`` stream while ``L`` computes.
    * ``compute(L)``: one barrier whose duration comes from the timing
      provider; it depends on ``memory(L)`` and ``compute(L - 1)``.
    * KV writes of the layer's newly processed tokens depend on ``compute(L)``.

    MoE splits compute into attention/router and FFN phases. Only known
    weights may prefetch across layers; selected experts wait for this
    layer's routing phase, even when a router trace was supplied offline.

    The tail (final norm, LM head when any request emits, tail compute) follows
    the last layer's compute, and ``batch/complete`` joins the tail with every
    KV write.
    """

    def __init__(
        self,
        *,
        models: Mapping[str, ModelSpec],
        request_trace: RequestTrace,
        router: RouterProvider | None = None,
        timing: TimingProvider | None = None,
        prefetch_depth: int | None = None,
    ) -> None:
        if not models:
            raise HBServeError("serving compiler requires models")
        self.models = dict(models)
        for model_id, model in self.models.items():
            if model_id != model.model_id:
                raise HBServeError(
                    "model map key differs from the ModelSpec model_id"
                )
        calibrated = timing is not None and timing.timing_model == "gpu_calibrated"
        if prefetch_depth is None:
            prefetch_depth = 0 if calibrated else DEFAULT_PREFETCH_DEPTH
        if calibrated and prefetch_depth != 0:
            raise HBServeError("GPU operator calibration requires measured serial kernel order (prefetch_depth=0)")
        if (
            isinstance(prefetch_depth, bool)
            or not isinstance(prefetch_depth, int)
            or prefetch_depth < 0
        ):
            raise HBServeError("prefetch_depth must be an integer >= 0")
        self.prefetch_depth = prefetch_depth
        self.request_trace = request_trace
        self.requests = {
            request.request_id: request for request in request_trace.requests
        }
        self.router = router
        self.timing: TimingProvider = (
            timing if timing is not None else MemoryOnlyTimingProvider()
        )
        any_moe = False
        for request in request_trace.requests:
            try:
                model = self.models[request.model_id]
            except KeyError as error:
                raise HBServeError(
                    f"request {request.request_id} names unknown model "
                    f"{request.model_id}"
                ) from error
            if request.token_ids is not None and any(
                token >= model.vocab_size for token in request.token_ids
            ):
                raise HBServeError(
                    f"request {request.request_id} contains an out-of-vocabulary "
                    "token ID"
                )
            any_moe = any_moe or any(layer.is_moe for layer in model.layers)
        if any_moe and router is None:
            raise HBServeError("MoE workloads require a router provider")
        if not any_moe and router is not None:
            raise HBServeError(
                "a router provider was supplied for an all-dense workload"
            )
        if router is not None:
            router.validate_complete(
                requests=request_trace.requests,
                models=self.models,
            )

    @staticmethod
    def _surrogate_token(model: ModelSpec, request: RequestSpec, token: int) -> int:
        key = (
            f"{model.model_id}\0{request.request_id}\0{token}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (
            model.vocab_size
        )

    def _token_id(
        self, model: ModelSpec, request: RequestSpec, token_index: int
    ) -> tuple[int, str]:
        if token_index >= request.processed_input_tokens:
            raise HBServeError(
                f"request {request.request_id} token index exceeds its processed "
                "input-token lifetime"
            )
        if request.token_ids is not None:
            return request.token_ids[token_index], "trace_token_id"
        return (
            self._surrogate_token(model, request, token_index),
            "sha256_surrogate_token_id",
        )

    def _validate_schedule(
        self, batch: ScheduledBatch
    ) -> tuple[ModelSpec, tuple[tuple[BatchSlice, RequestSpec], ...]]:
        try:
            model = self.models[batch.model_id]
        except KeyError as error:
            raise HBServeError(
                f"scheduled batch names unknown model {batch.model_id}"
            ) from error
        rows: list[tuple[BatchSlice, RequestSpec]] = []
        for batch_slice in batch.slices:
            try:
                request = self.requests[batch_slice.request_id]
            except KeyError as error:
                raise HBServeError(
                    f"scheduled batch names unknown request "
                    f"{batch_slice.request_id}"
                ) from error
            if request.model_id != model.model_id:
                raise HBServeError(
                    f"request {request.request_id} is batched under a different "
                    "model"
                )
            end = batch_slice.token_end
            if end > request.processed_input_tokens:
                raise HBServeError(
                    f"slice exceeds request {request.request_id} processed "
                    "input-token lifetime"
                )
            if batch_slice.phase == "decode":
                if batch_slice.token_begin < request.prompt_tokens:
                    raise HBServeError(
                        f"decode slice of request {request.request_id} starts "
                        "inside the prompt"
                    )
            elif batch_slice.emits_output and end < request.prompt_tokens:
                raise HBServeError(
                    f"prefill slice of request {request.request_id} emits "
                    "output before the prompt is complete"
                )
            rows.append((batch_slice, request))
        return model, tuple(rows)

    def compile(self, batch: ScheduledBatch) -> CanonicalServingBatch:
        if self.timing.timing_model == "gpu_calibrated":
            from hbserve.gpu_compiler import compile_calibrated
            return compile_calibrated(self, batch)
        model, rows = self._validate_schedule(batch)
        timing = self.timing.timing_for(model=model, batch=batch)
        if len(timing.layer_ns) != model.num_layers:
            raise HBServeError(
                "timing provider layer count differs from the model"
            )

        operations: list[SemanticOperation] = []
        audit: dict[str, dict[str, Any]] = {}
        prefix = f"serve/b{batch.batch_id}"
        next_memory = 0
        next_barrier = 0

        def barrier(
            *, role: str, dependencies: Iterable[str], duration_ns: float = 0.0
        ) -> str:
            nonlocal next_barrier
            identifier = f"{prefix}/d{next_barrier}"
            next_barrier += 1
            unique = tuple(dict.fromkeys(dependencies))
            operation = SemanticOperation(
                id=identifier,
                op=None,
                object_id=None,
                offset=0,
                bytes=0,
                duration_ns=duration_ns,
                dependencies=unique,
                role=role,
            )
            operations.append(operation)
            audit[identifier] = {
                "role": role,
                "source": "dependency_or_compute",
                "duration_ns": duration_ns,
            }
            return identifier

        object_by_id = model.object_by_id

        def memory(
            *,
            role: str,
            object_id: str,
            offset: int,
            byte_count: int,
            op: str,
            dependencies: Iterable[str],
            labels: Mapping[str, Any],
            object_size: int | None = None,
        ) -> str:
            nonlocal next_memory
            identifier = f"{prefix}/m{next_memory}"
            next_memory += 1
            if object_size is None:
                try:
                    memory_object = object_by_id[object_id]
                except KeyError as error:
                    raise HBServeError(
                        f"unknown model memory object: {object_id}"
                    ) from error
                object_size = memory_object.bytes
            if offset < 0 or byte_count <= 0 or offset + byte_count > object_size:
                raise HBServeError(
                    f"operation {identifier} escapes memory object {object_id}"
                )
            operation = SemanticOperation(
                id=identifier,
                op=op,
                object_id=object_id,
                offset=offset,
                bytes=byte_count,
                duration_ns=0.0,
                dependencies=tuple(dict.fromkeys(dependencies)),
                role=role,
            )
            operations.append(operation)
            audit[identifier] = {"role": role, **dict(labels)}
            return identifier

        start = barrier(role="batch/start", dependencies=())

        embedding = object_by_id[model.object_id("embedding")]
        embedding_ops: list[str] = []
        for batch_slice, request in rows:
            for token_index in range(batch_slice.token_begin, batch_slice.token_end):
                token_id, token_source = self._token_id(
                    model, request, token_index
                )
                embedding_ops.append(
                    memory(
                        role="embedding/read",
                        object_id=embedding.id,
                        offset=token_id * model.embedding_row_bytes,
                        byte_count=model.embedding_row_bytes,
                        op="R",
                        dependencies=(start,),
                        labels={
                            "request_id": request.request_id,
                            "token_index": token_index,
                            "token_id": token_id,
                            "token_id_source": token_source,
                        },
                    )
                )
        embedded = barrier(role="embedding/complete", dependencies=embedding_ops)

        compute_barriers: list[str] = []
        kv_writes: list[str] = []
        for layer_id, layer in enumerate(model.layers):
            # The layer buffer that L reuses frees when compute(L-1-depth)
            # finishes; earlier layers only wait for the embedding gather.
            release_layer = layer_id - 1 - self.prefetch_depth
            memory_root = (
                compute_barriers[release_layer] if release_layer >= 0 else embedded
            )
            memory_ops: list[str] = []
            if layer.attention_weight_bytes:
                memory_ops.append(
                    memory(
                        role="attention/weights",
                        object_id=model.object_id("attention", layer_id),
                        offset=0,
                        byte_count=layer.attention_weight_bytes,
                        op="R",
                        dependencies=(memory_root,),
                        labels={"layer": layer_id},
                    )
                )
            layer_writes = []
            for batch_slice, request in rows:
                layer_writes.append(
                    memory(
                        role="attention/kv_write",
                        object_id=model.kv_object_id(request.request_id, layer_id),
                        offset=batch_slice.token_begin * layer.kv_bytes_per_token,
                        byte_count=batch_slice.token_count * layer.kv_bytes_per_token,
                        op="W",
                        dependencies=(compute_barriers[-1] if compute_barriers else embedded,),
                        labels={
                            "request_id": request.request_id,
                            "layer": layer_id,
                            "token_begin": batch_slice.token_begin,
                            "token_count": batch_slice.token_count,
                        },
                        object_size=(
                            request.processed_input_tokens
                            * layer.kv_bytes_per_token
                        ),
                    )
                )

            kv_writes.extend(layer_writes)
            for batch_slice, request in rows:
                if not batch_slice.context_tokens_before:
                    continue
                memory_ops.append(
                    memory(
                        role="attention/kv_read",
                        object_id=model.kv_object_id(request.request_id, layer_id),
                        offset=0,
                        byte_count=(
                            batch_slice.context_tokens_before
                            * layer.kv_bytes_per_token
                        ),
                        op="R",
                        dependencies=(memory_root, *layer_writes),
                        labels={
                            "request_id": request.request_id,
                            "layer": layer_id,
                            "context_tokens": batch_slice.context_tokens_before,
                            "traffic_semantics": (
                                "existing_context_once_io_optimal"
                            ),
                        },
                        object_size=(
                            request.processed_input_tokens
                            * layer.kv_bytes_per_token
                        ),
                    )
                )
            if layer.ffn_weight_bytes:
                memory_ops.append(
                    memory(
                        role="ffn/weights",
                        object_id=model.object_id("ffn", layer_id),
                        offset=0,
                        byte_count=layer.ffn_weight_bytes,
                        op="R",
                        dependencies=(memory_root,),
                        labels={"layer": layer_id},
                    )
                )
            if layer.router_weight_bytes:
                memory_ops.append(
                    memory(
                        role="moe/router_weights",
                        object_id=model.object_id("router", layer_id),
                        offset=0,
                        byte_count=layer.router_weight_bytes,
                        op="R",
                        dependencies=(memory_root,),
                        labels={"layer": layer_id},
                    )
                )
            routing = None
            routing_ns = 0.0
            if layer.is_moe:
                if not timing.routing_ns and timing.layer_ns[layer_id] != 0.0:
                    raise HBServeError(
                        "MoE compute timing requires an explicit routing phase"
                    )
                routing_ns = timing.routing_ns[layer_id] if timing.routing_ns else 0.0
                routing = barrier(
                    role=f"layer/{layer_id}/routing_ready",
                    dependencies=(
                        *memory_ops,
                        compute_barriers[-1] if compute_barriers else embedded,
                    ),
                    duration_ns=routing_ns,
                )
            if layer.shared_expert_weight_bytes:
                memory_ops.append(
                    memory(
                        role="moe/shared_expert_weights",
                        object_id=model.object_id("shared_expert", layer_id),
                        offset=0,
                        byte_count=layer.shared_expert_weight_bytes,
                        op="R",
                        dependencies=(memory_root,),
                        labels={"layer": layer_id},
                    )
                )
            if layer.is_moe:
                if self.router is None:
                    raise HBServeError(
                        "MoE batch reached the compiler without a router"
                    )
                token_counts: dict[int, int] = {}
                for batch_slice, request in rows:
                    for token_index in range(
                        batch_slice.token_begin, batch_slice.token_end
                    ):
                        experts = self.router.experts_for(
                            request=request,
                            token_index=token_index,
                            layer=layer_id,
                            model=model,
                        )
                        for expert in experts:
                            token_counts[expert] = token_counts.get(expert, 0) + 1
                for expert_id in sorted(token_counts):
                    memory_ops.append(
                        memory(
                            role="moe/routed_expert_weights",
                            object_id=model.expert_object_id(layer_id, expert_id),
                            offset=0,
                            byte_count=layer.expert_weight_bytes[expert_id],
                            op="R",
                            dependencies=(routing,),
                            labels={
                                "layer": layer_id,
                                "expert": expert_id,
                                "routed_tokens": token_counts[expert_id],
                                "traffic_semantics": (
                                    "unique_expert_once_per_batch_layer"
                                ),
                            },
                        )
                    )
            memory_complete = barrier(
                role=f"layer/{layer_id}/memory_complete",
                dependencies=memory_ops if memory_ops else (memory_root,),
            )
            compute_dependencies = [memory_complete]
            if compute_barriers:
                compute_dependencies.append(compute_barriers[-1])
            compute = barrier(
                role=f"layer/{layer_id}/compute",
                dependencies=compute_dependencies,
                duration_ns=timing.layer_ns[layer_id] - routing_ns,
            )
            compute_barriers.append(compute)

        final_norm = object_by_id[model.object_id("final_norm")]
        tail = memory(
            role="final_norm/read",
            object_id=final_norm.id,
            offset=0,
            byte_count=final_norm.bytes,
            op="R",
            dependencies=(compute_barriers[-1],),
            labels={},
        )
        if batch.emits_output:
            head = object_by_id[model.lm_head_object_id()]
            tail = memory(
                role="lm_head/read",
                object_id=head.id,
                offset=0,
                byte_count=head.bytes,
                op="R",
                dependencies=(tail,),
                labels={
                    "output_requests": sum(
                        item.emits_output for item, _ in rows
                    )
                },
            )
        tail_compute = barrier(
            role="tail/compute",
            dependencies=(tail,),
            duration_ns=timing.tail_ns,
        )
        barrier(role="batch/complete", dependencies=(tail_compute, *kv_writes))

        uses_moe = any(layer.is_moe for layer in model.layers)
        return CanonicalServingBatch(
            schedule=batch,
            model_sha256=model.digest,
            request_trace_sha256=self.request_trace.digest,
            router_trace_sha256=(
                self.router.digest if uses_moe and self.router is not None else None
            ),
            timing_model=timing.timing_model,
            timing_evidence_state=timing.evidence_state,
            prefetch_depth=self.prefetch_depth,
            operations=tuple(operations),
            audit=audit,
        )


def kv_objects_for_request(
    model: ModelSpec, request: RequestSpec
) -> tuple[MemoryObject, ...]:
    """Return the logical per-layer KV objects spanning one request's lifetime.

    The object size is the request's whole ``P + O - 1`` token context; the
    placement backs it with paged blocks that grow as tokens are processed.
    """

    if request.model_id != model.model_id:
        raise HBServeError("request/model mismatch for KV objects")
    result: list[MemoryObject] = []
    for layer_id, layer in enumerate(model.layers):
        result.append(
            MemoryObject(
                id=model.kv_object_id(request.request_id, layer_id),
                model_id=model.model_id,
                kind="kv",
                bytes=(
                    request.processed_input_tokens * layer.kv_bytes_per_token
                ),
                mutable=True,
                layer=layer_id,
            )
        )
    return tuple(result)
