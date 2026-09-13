#!/usr/bin/env python3
"""Strict contracts for HBServe request-driven LLM workloads.

These classes describe *derived* memory demand and *modeled* compute time.
They do not claim that the result is a measured GPU load/store trace.  The
fixed traffic and compute semantics are explicit so that a later kernel/tile
frontend can replace them without silently changing the meaning of an
experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import hashlib
import json
import math
from typing import Any, Mapping, Protocol, Sequence


UINT64_MAX = 2**64 - 1
MODEL_SCHEMA = {"name": "hbserve.model", "version": 2}
REQUEST_TRACE_SCHEMA = {
    "name": "hbserve.request_trace",
    "version": 1,
}
ROUTER_TRACE_SCHEMA = {
    "name": "hbserve.router_trace",
    "version": 1,
}
CANONICAL_BATCH_SCHEMA = {
    "name": "hbserve.canonical_batch",
    "version": 2,
}

TRAFFIC_MODEL = {
    "weight_reads": "once_per_batch_per_object",
    "expert_weight_reads": "once_per_unique_expert_per_batch_layer",
    "attention_kv_reads": "existing_context_once_per_request_per_layer",
    "kv_writes": "scheduled_input_tokens_once_per_layer",
    "embedding_reads": "one_row_per_scheduled_input_token",
    "output_head_reads": "once_per_batch_that_emits_output",
    "offchip_scratch": "excluded_without_kernel_trace",
    "output_token_semantics": "first_token_from_final_prefill_v1",
}
# FLOP accounting that the roofline timing provider consumes.  Linear FLOPs
# are 2 x the matrix parameters a token activates in the layer (attention
# projections plus the dense FFN, or router + shared + top_k routed experts
# for MoE).  Attention score/value FLOPs are 4 x heads x head_dim per
# scheduled token per context token (2 for QK^T, 2 for PV; MLA uses the
# absorbed QK width and V width), summed causally over the tokens a slice
# processes.  The LM head costs 2 x vocab x hidden per output-emitting request.
COMPUTE_MODEL = {
    "linear_flops": "2_x_activated_matrix_parameters_per_token_per_layer",
    "attention_flops": (
        "4_x_heads_x_head_dim_per_token_per_context_token_causal"
    ),
    "lm_head_flops": "2_x_vocab_x_hidden_per_output_emitting_request",
}
MODEL_PROVENANCE_KINDS = {
    "checkpoint_manifest",
    "published_descriptor",
    "synthetic_sensitivity",
    "ci_fixture",
}
TIMING_MODELS = {"memory_only", "linear", "roofline"}
TIMING_EVIDENCE_STATES = {
    "memory_only",
    "modeled_sensitivity",
    "modeled_roofline",
    "calibrated",
}
SLICE_PHASES = {"prefill", "decode"}


class HBServeError(ValueError):
    """A HBServe workload violates a causal or accounting contract."""


def canonical_sha256(value: Any) -> str:
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256()
    chunks: list[str] = []
    size = 0
    for chunk in encoder.iterencode(value):
        chunks.append(chunk)
        size += len(chunk)
        if size >= 65536:
            digest.update("".join(chunks).encode("utf-8"))
            chunks.clear()
            size = 0
    if chunks:
        digest.update("".join(chunks).encode("utf-8"))
    return digest.hexdigest()


def _integer(value: Any, description: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise HBServeError(
            f"{description} must be an integer >= {minimum}"
        )
    if value > UINT64_MAX:
        raise HBServeError(f"{description} exceeds uint64")
    return value


def _finite(value: Any, description: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HBServeError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise HBServeError(
            f"{description} must be finite and >= {minimum}"
        )
    return result


def _identifier(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value or any(
        not (
            character.isascii()
            and (character.isalnum() or character in "_-.:/")
        )
        for character in value
    ):
        raise HBServeError(
            f"{description} is not a non-empty protocol-safe identifier"
        )
    return value


def _entity_identifier(value: Any, description: str) -> str:
    result = _identifier(value, description)
    if "/" in result:
        raise HBServeError(
            f"{description} cannot contain '/' because it delimits object IDs"
        )
    return result


def _checked_add(lhs: int, rhs: int, description: str) -> int:
    _integer(lhs, f"{description} lhs")
    _integer(rhs, f"{description} rhs")
    if rhs > UINT64_MAX - lhs:
        raise HBServeError(f"{description} overflows uint64")
    return lhs + rhs


def _checked_mul(lhs: int, rhs: int, description: str) -> int:
    _integer(lhs, f"{description} lhs")
    _integer(rhs, f"{description} rhs")
    if lhs and rhs > UINT64_MAX // lhs:
        raise HBServeError(f"{description} overflows uint64")
    return lhs * rhs


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], description: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise HBServeError(
            f"{description} fields differ: missing={missing}, extra={extra}"
        )


def _sha256_or_none(value: Any, description: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HBServeError(f"{description} SHA-256 is malformed")
    return value


@dataclass(frozen=True)
class TraceProvenance:
    """Evidence boundary for request or router input."""

    kind: str
    source: str
    sha256: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {
            "production_trace",
            "model_generated_trace",
            "synthetic_sensitivity",
            "ci_fixture",
        }:
            raise HBServeError(
                f"unsupported trace provenance kind: {self.kind!r}"
            )
        if not isinstance(self.source, str) or not self.source:
            raise HBServeError("trace provenance source is empty")
        _sha256_or_none(self.sha256, "trace provenance")
        if self.kind in {"production_trace", "model_generated_trace"} and (
            self.sha256 is None
        ):
            raise HBServeError(
                f"{self.kind} provenance requires a source SHA-256"
            )

    def canonical(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source": self.source,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class LayerSpec:
    """One decoder layer's exact off-chip weight, KV, and FLOP ledger."""

    attention_weight_bytes: int
    ffn_weight_bytes: int
    router_weight_bytes: int
    shared_expert_weight_bytes: int
    expert_weight_bytes: tuple[int, ...]
    top_k: int
    kv_bytes_per_token: int
    flops_per_token: int
    attention_flops_per_context_token: int
    pre_routing_flops_per_token: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("attention_weight_bytes", self.attention_weight_bytes),
            ("ffn_weight_bytes", self.ffn_weight_bytes),
            ("router_weight_bytes", self.router_weight_bytes),
            ("shared_expert_weight_bytes", self.shared_expert_weight_bytes),
            ("flops_per_token", self.flops_per_token),
            (
                "attention_flops_per_context_token",
                self.attention_flops_per_context_token,
            ),
        ):
            _integer(value, f"layer {name}")
        _integer(self.kv_bytes_per_token, "layer kv_bytes_per_token", minimum=1)
        if self.pre_routing_flops_per_token is not None:
            _integer(self.pre_routing_flops_per_token, "layer pre_routing_flops_per_token")
            if (
                not self.is_moe
                or self.pre_routing_flops_per_token > self.flops_per_token
            ):
                raise HBServeError(
                    "pre-routing FLOPs require MoE and cannot exceed layer FLOPs"
                )
        for index, value in enumerate(self.expert_weight_bytes):
            _integer(value, f"expert_weight_bytes[{index}]", minimum=1)
        if self.expert_weight_bytes:
            _integer(self.top_k, "layer top_k", minimum=1)
            if self.top_k > len(self.expert_weight_bytes):
                raise HBServeError(
                    "layer top_k exceeds the routed expert count"
                )
            if self.router_weight_bytes == 0:
                raise HBServeError(
                    "a routed-expert layer must account for router weights"
                )
        elif self.top_k != 0:
            raise HBServeError(
                "a layer without routed experts must have top_k=0"
            )
        elif self.router_weight_bytes != 0:
            raise HBServeError(
                "a layer without routed experts cannot have router weights"
            )
        if (
            self.attention_weight_bytes == 0
            and self.ffn_weight_bytes == 0
            and self.router_weight_bytes == 0
            and self.shared_expert_weight_bytes == 0
            and not self.expert_weight_bytes
        ):
            raise HBServeError("a layer must contain some model weights")

    @property
    def is_moe(self) -> bool:
        return bool(self.expert_weight_bytes)

    @property
    def weight_bytes(self) -> int:
        total = _checked_add(
            self.attention_weight_bytes,
            self.ffn_weight_bytes,
            "layer dense weight bytes",
        )
        total = _checked_add(
            total,
            self.router_weight_bytes,
            "layer router weight bytes",
        )
        total = _checked_add(
            total,
            self.shared_expert_weight_bytes,
            "layer shared weight bytes",
        )
        for expert_bytes in self.expert_weight_bytes:
            total = _checked_add(total, expert_bytes, "layer expert weight bytes")
        return total

    def canonical(self) -> dict[str, Any]:
        return {
            "attention_weight_bytes": self.attention_weight_bytes,
            "ffn_weight_bytes": self.ffn_weight_bytes,
            "router_weight_bytes": self.router_weight_bytes,
            "shared_expert_weight_bytes": self.shared_expert_weight_bytes,
            "expert_weight_bytes": list(self.expert_weight_bytes),
            "top_k": self.top_k,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "flops_per_token": self.flops_per_token,
            "attention_flops_per_context_token": (
                self.attention_flops_per_context_token
            ),
            **({"pre_routing_flops_per_token": self.pre_routing_flops_per_token}
               if self.pre_routing_flops_per_token is not None else {}),
        }


LAYER_FIELDS = {
    "attention_weight_bytes",
    "ffn_weight_bytes",
    "router_weight_bytes",
    "shared_expert_weight_bytes",
    "expert_weight_bytes",
    "top_k",
    "kv_bytes_per_token",
    "flops_per_token",
    "attention_flops_per_context_token",
}


@dataclass(frozen=True)
class MemoryObject:
    id: str
    model_id: str
    kind: str
    bytes: int
    mutable: bool
    layer: int | None = None
    expert: int | None = None

    def __post_init__(self) -> None:
        _identifier(self.id, "memory object id")
        _entity_identifier(self.model_id, "memory object model id")
        if self.kind not in {
            "embedding",
            "attention_weight",
            "ffn_weight",
            "router_weight",
            "shared_expert_weight",
            "expert_weight",
            "final_norm",
            "lm_head",
            "kv",
        }:
            raise HBServeError(
                f"unsupported memory object kind: {self.kind!r}"
            )
        _integer(self.bytes, f"memory object {self.id} bytes", minimum=1)
        if not isinstance(self.mutable, bool):
            raise HBServeError(
                f"memory object {self.id} mutable must be boolean"
            )
        if self.layer is not None:
            _integer(self.layer, f"memory object {self.id} layer")
        if self.expert is not None:
            _integer(self.expert, f"memory object {self.id} expert")
        if self.kind == "expert_weight" and (
            self.layer is None or self.expert is None
        ):
            raise HBServeError(
                "expert weight objects require layer and expert IDs"
            )
        if self.kind == "kv" and not self.mutable:
            raise HBServeError("KV objects must be mutable")
        if self.kind != "kv" and self.mutable:
            raise HBServeError("model-weight objects must be immutable")

    def canonical(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model_id": self.model_id,
            "kind": self.kind,
            "bytes": self.bytes,
            "mutable": self.mutable,
            "layer": self.layer,
            "expert": self.expert,
        }


@dataclass(frozen=True)
class ModelSpec:
    """A model's explicit memory objects, without inferred parameter sizes."""

    model_id: str
    provenance: Mapping[str, Any]
    vocab_size: int
    embedding_bytes: int
    final_norm_bytes: int
    lm_head_bytes: int
    tie_word_embeddings: bool
    layers: tuple[LayerSpec, ...]
    lm_head_flops_per_token: int

    def __post_init__(self) -> None:
        _entity_identifier(self.model_id, "model id")
        if not isinstance(self.provenance, Mapping):
            raise HBServeError("model provenance must be an object")
        _exact_keys(
            self.provenance,
            {"kind", "source", "sha256"},
            "model provenance",
        )
        provenance_kind = self.provenance.get("kind")
        provenance_source = self.provenance.get("source")
        provenance_sha = _sha256_or_none(
            self.provenance.get("sha256"), "model provenance"
        )
        if provenance_kind not in MODEL_PROVENANCE_KINDS:
            raise HBServeError(
                f"unsupported model provenance kind: {provenance_kind!r}"
            )
        if not isinstance(provenance_source, str) or not provenance_source:
            raise HBServeError("model provenance source is empty")
        if provenance_kind in {
            "checkpoint_manifest",
            "published_descriptor",
        } and provenance_sha is None:
            raise HBServeError(
                f"{provenance_kind} model provenance requires a source SHA-256"
            )
        _integer(self.vocab_size, "model vocab_size", minimum=1)
        _integer(self.embedding_bytes, "model embedding_bytes", minimum=1)
        _integer(self.final_norm_bytes, "model final_norm_bytes", minimum=1)
        _integer(self.lm_head_bytes, "model lm_head_bytes", minimum=1)
        _integer(self.lm_head_flops_per_token, "model lm_head_flops_per_token")
        if self.embedding_bytes % self.vocab_size:
            raise HBServeError(
                "embedding bytes must divide into integral vocabulary rows"
            )
        if self.tie_word_embeddings and self.lm_head_bytes != self.embedding_bytes:
            raise HBServeError(
                "a tied LM head must have the same size as the embedding"
            )
        if not self.layers:
            raise HBServeError("model must contain at least one layer")
        if not all(isinstance(layer, LayerSpec) for layer in self.layers):
            raise HBServeError("model layers must be LayerSpec values")
        # Force checked aggregate accounting during construction.
        _ = self.weight_footprint_bytes

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def embedding_row_bytes(self) -> int:
        return self.embedding_bytes // self.vocab_size

    def object_id(self, component: str, layer: int | None = None) -> str:
        if layer is None:
            return f"model/{self.model_id}/{component}"
        return f"model/{self.model_id}/layer/{layer}/{component}"

    def expert_object_id(self, layer: int, expert: int) -> str:
        return self.object_id(f"expert/{expert}", layer)

    def kv_object_id(self, request_id: str, layer: int) -> str:
        _entity_identifier(request_id, "KV request id")
        _integer(layer, "KV layer")
        if layer >= self.num_layers:
            raise HBServeError("KV layer exceeds model layer count")
        return f"request/{request_id}/model/{self.model_id}/layer/{layer}/kv"

    @cached_property
    def memory_objects(self) -> tuple[MemoryObject, ...]:
        objects: list[MemoryObject] = [
            MemoryObject(
                id=self.object_id("embedding"),
                model_id=self.model_id,
                kind="embedding",
                bytes=self.embedding_bytes,
                mutable=False,
            )
        ]
        for layer_id, layer in enumerate(self.layers):
            if layer.attention_weight_bytes:
                objects.append(
                    MemoryObject(
                        id=self.object_id("attention", layer_id),
                        model_id=self.model_id,
                        kind="attention_weight",
                        bytes=layer.attention_weight_bytes,
                        mutable=False,
                        layer=layer_id,
                    )
                )
            if layer.ffn_weight_bytes:
                objects.append(
                    MemoryObject(
                        id=self.object_id("ffn", layer_id),
                        model_id=self.model_id,
                        kind="ffn_weight",
                        bytes=layer.ffn_weight_bytes,
                        mutable=False,
                        layer=layer_id,
                    )
                )
            if layer.router_weight_bytes:
                objects.append(
                    MemoryObject(
                        id=self.object_id("router", layer_id),
                        model_id=self.model_id,
                        kind="router_weight",
                        bytes=layer.router_weight_bytes,
                        mutable=False,
                        layer=layer_id,
                    )
                )
            if layer.shared_expert_weight_bytes:
                objects.append(
                    MemoryObject(
                        id=self.object_id("shared_expert", layer_id),
                        model_id=self.model_id,
                        kind="shared_expert_weight",
                        bytes=layer.shared_expert_weight_bytes,
                        mutable=False,
                        layer=layer_id,
                    )
                )
            for expert_id, expert_bytes in enumerate(layer.expert_weight_bytes):
                objects.append(
                    MemoryObject(
                        id=self.expert_object_id(layer_id, expert_id),
                        model_id=self.model_id,
                        kind="expert_weight",
                        bytes=expert_bytes,
                        mutable=False,
                        layer=layer_id,
                        expert=expert_id,
                    )
                )
        objects.append(
            MemoryObject(
                id=self.object_id("final_norm"),
                model_id=self.model_id,
                kind="final_norm",
                bytes=self.final_norm_bytes,
                mutable=False,
            )
        )
        if not self.tie_word_embeddings:
            objects.append(
                MemoryObject(
                    id=self.object_id("lm_head"),
                    model_id=self.model_id,
                    kind="lm_head",
                    bytes=self.lm_head_bytes,
                    mutable=False,
                )
            )
        identifiers = [item.id for item in objects]
        if len(identifiers) != len(set(identifiers)):
            raise HBServeError("model memory object IDs are not unique")
        return tuple(objects)

    @cached_property
    def object_by_id(self) -> Mapping[str, MemoryObject]:
        return {item.id: item for item in self.memory_objects}

    @cached_property
    def weight_footprint_bytes(self) -> int:
        total = 0
        for memory_object in self.memory_objects:
            total = _checked_add(
                total,
                memory_object.bytes,
                f"model {self.model_id} weight footprint",
            )
        return total

    @cached_property
    def kv_bytes_per_token(self) -> int:
        """KV bytes one token occupies across every layer."""

        total = 0
        for layer in self.layers:
            total = _checked_add(
                total, layer.kv_bytes_per_token, "model KV bytes per token"
            )
        return total

    def lm_head_object_id(self) -> str:
        return self.object_id(
            "embedding" if self.tie_word_embeddings else "lm_head"
        )

    def canonical(self) -> dict[str, Any]:
        return {
            "schema": MODEL_SCHEMA,
            "model_id": self.model_id,
            "provenance": dict(self.provenance),
            "vocab_size": self.vocab_size,
            "embedding_bytes": self.embedding_bytes,
            "final_norm_bytes": self.final_norm_bytes,
            "lm_head_bytes": self.lm_head_bytes,
            "lm_head_flops_per_token": self.lm_head_flops_per_token,
            "tie_word_embeddings": self.tie_word_embeddings,
            "layers": [layer.canonical() for layer in self.layers],
            "traffic_model": dict(TRAFFIC_MODEL),
            "compute_model": dict(COMPUTE_MODEL),
        }

    @cached_property
    def digest(self) -> str:
        return canonical_sha256(self.canonical())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelSpec":
        _exact_keys(
            value,
            {
                "schema",
                "model_id",
                "provenance",
                "vocab_size",
                "embedding_bytes",
                "final_norm_bytes",
                "lm_head_bytes",
                "lm_head_flops_per_token",
                "tie_word_embeddings",
                "layers",
                "traffic_model",
                "compute_model",
            },
            "model descriptor",
        )
        if value.get("schema") != MODEL_SCHEMA:
            raise HBServeError(
                "unsupported serving model schema; expected "
                f"{MODEL_SCHEMA['name']} v{MODEL_SCHEMA['version']}"
            )
        if value.get("traffic_model") != TRAFFIC_MODEL:
            raise HBServeError(
                "serving model traffic semantics differ from schema v2"
            )
        if value.get("compute_model") != COMPUTE_MODEL:
            raise HBServeError(
                "serving model compute semantics differ from schema v2"
            )
        raw_layers = value.get("layers")
        if not isinstance(raw_layers, list) or not raw_layers:
            raise HBServeError("model layers must be non-empty")
        layers: list[LayerSpec] = []
        for layer_index, raw_layer in enumerate(raw_layers):
            if not isinstance(raw_layer, Mapping):
                raise HBServeError(
                    f"layers[{layer_index}] must be an object"
                )
            _exact_keys(
                raw_layer,
                LAYER_FIELDS | ({"pre_routing_flops_per_token"}
                                if "pre_routing_flops_per_token" in raw_layer else set()),
                f"layers[{layer_index}]",
            )
            raw_experts = raw_layer.get("expert_weight_bytes")
            if not isinstance(raw_experts, list):
                raise HBServeError(
                    f"layers[{layer_index}].expert_weight_bytes must be an array"
                )
            layer = LayerSpec(
                attention_weight_bytes=_integer(
                    raw_layer.get("attention_weight_bytes"),
                    f"layers[{layer_index}] attention bytes",
                ),
                ffn_weight_bytes=_integer(
                    raw_layer.get("ffn_weight_bytes"),
                    f"layers[{layer_index}] FFN bytes",
                ),
                router_weight_bytes=_integer(
                    raw_layer.get("router_weight_bytes"),
                    f"layers[{layer_index}] router bytes",
                ),
                shared_expert_weight_bytes=_integer(
                    raw_layer.get("shared_expert_weight_bytes"),
                    f"layers[{layer_index}] shared expert bytes",
                ),
                expert_weight_bytes=tuple(
                    _integer(
                        expert_bytes,
                        f"layers[{layer_index}].expert_weight_bytes[{expert}]",
                        minimum=1,
                    )
                    for expert, expert_bytes in enumerate(raw_experts)
                ),
                top_k=_integer(
                    raw_layer.get("top_k"),
                    f"layers[{layer_index}] top_k",
                ),
                kv_bytes_per_token=_integer(
                    raw_layer.get("kv_bytes_per_token"),
                    f"layers[{layer_index}] KV bytes/token",
                    minimum=1,
                ),
                flops_per_token=_integer(
                    raw_layer.get("flops_per_token"),
                    f"layers[{layer_index}] flops_per_token",
                ),
                attention_flops_per_context_token=_integer(
                    raw_layer.get("attention_flops_per_context_token"),
                    f"layers[{layer_index}] attention_flops_per_context_token",
                ),
                pre_routing_flops_per_token=(
                    _integer(raw_layer["pre_routing_flops_per_token"],
                             f"layers[{layer_index}] pre_routing_flops_per_token")
                    if "pre_routing_flops_per_token" in raw_layer else None
                ),
            )
            layers.append(layer)
        provenance = value.get("provenance")
        if not isinstance(provenance, Mapping):
            raise HBServeError("model provenance must be an object")
        tied = value.get("tie_word_embeddings")
        if not isinstance(tied, bool):
            raise HBServeError("tie_word_embeddings must be boolean")
        return cls(
            model_id=_entity_identifier(value.get("model_id"), "model id"),
            provenance=dict(provenance),
            vocab_size=_integer(value.get("vocab_size"), "vocab_size", minimum=1),
            embedding_bytes=_integer(
                value.get("embedding_bytes"), "embedding_bytes", minimum=1
            ),
            final_norm_bytes=_integer(
                value.get("final_norm_bytes"), "final_norm_bytes", minimum=1
            ),
            lm_head_bytes=_integer(
                value.get("lm_head_bytes"), "lm_head_bytes", minimum=1
            ),
            tie_word_embeddings=tied,
            layers=tuple(layers),
            lm_head_flops_per_token=_integer(
                value.get("lm_head_flops_per_token"), "lm_head_flops_per_token"
            ),
        )


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    arrival_ns: float
    model_id: str
    prompt_tokens: int
    output_tokens: int
    token_ids: tuple[int, ...] | None = None
    cache_salt: str = ""

    def __post_init__(self) -> None:
        _entity_identifier(self.request_id, "request id")
        _entity_identifier(self.model_id, "request model id")
        if not isinstance(self.cache_salt, str):
            raise HBServeError("request cache_salt must be a string")
        _finite(self.arrival_ns, f"request {self.request_id} arrival_ns")
        _integer(
            self.prompt_tokens,
            f"request {self.request_id} prompt_tokens",
            minimum=1,
        )
        _integer(
            self.output_tokens,
            f"request {self.request_id} output_tokens",
            minimum=1,
        )
        expected_tokens = self.processed_input_tokens
        if self.token_ids is not None:
            if len(self.token_ids) != expected_tokens:
                raise HBServeError(
                    f"request {self.request_id} token_ids must contain exactly "
                    f"{expected_tokens} processed input tokens"
                )
            for index, token in enumerate(self.token_ids):
                _integer(token, f"request {self.request_id} token_ids[{index}]")

    @property
    def processed_input_tokens(self) -> int:
        return (
            _checked_add(
                self.prompt_tokens,
                self.output_tokens - 1,
                f"request {self.request_id} processed token count",
            )
        )

    def canonical(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "arrival_ns": self.arrival_ns,
            "model_id": self.model_id,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "token_ids": None if self.token_ids is None else list(self.token_ids),
            "cache_salt": self.cache_salt,
        }


@dataclass(frozen=True)
class RequestTrace:
    provenance: TraceProvenance
    requests: tuple[RequestSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, TraceProvenance):
            raise HBServeError(
                "request trace provenance must be TraceProvenance"
            )
        if not self.requests:
            raise HBServeError("request trace must not be empty")
        identifiers: set[str] = set()
        prior_key: tuple[float, str] | None = None
        for request in self.requests:
            if not isinstance(request, RequestSpec):
                raise HBServeError("request trace contains a non-request")
            if request.request_id in identifiers:
                raise HBServeError(
                    f"duplicate request id: {request.request_id}"
                )
            key = (request.arrival_ns, request.request_id)
            if prior_key is not None and key < prior_key:
                raise HBServeError(
                    "request trace must be sorted by arrival_ns then request_id"
                )
            identifiers.add(request.request_id)
            prior_key = key

    def canonical(self) -> dict[str, Any]:
        return {
            "schema": REQUEST_TRACE_SCHEMA,
            "provenance": self.provenance.canonical(),
            "requests": [request.canonical() for request in self.requests],
        }

    @cached_property
    def digest(self) -> str:
        return canonical_sha256(self.canonical())


@dataclass(frozen=True)
class RouterDecision:
    request_id: str
    token_index: int
    layer: int
    experts: tuple[int, ...]

    def __post_init__(self) -> None:
        _entity_identifier(self.request_id, "router request id")
        _integer(self.token_index, "router token_index")
        _integer(self.layer, "router layer")
        if not self.experts:
            raise HBServeError("router decision must select experts")
        for position, expert in enumerate(self.experts):
            _integer(expert, f"router expert[{position}]")
        if len(self.experts) != len(set(self.experts)):
            raise HBServeError("router decision repeats an expert")

    @property
    def key(self) -> tuple[str, int, int]:
        return self.request_id, self.token_index, self.layer

    def canonical(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "token_index": self.token_index,
            "layer": self.layer,
            "experts": list(self.experts),
        }


class RouterProvider(Protocol):
    provenance: TraceProvenance

    @property
    def digest(self) -> str: ...

    def experts_for(
        self,
        *,
        request: RequestSpec,
        token_index: int,
        layer: int,
        model: ModelSpec,
    ) -> tuple[int, ...]: ...

    def validate_complete(
        self,
        *,
        requests: Sequence[RequestSpec],
        models: Mapping[str, ModelSpec],
    ) -> None: ...


@dataclass(frozen=True)
class RouterTrace:
    provenance: TraceProvenance
    decisions: tuple[RouterDecision, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, TraceProvenance):
            raise HBServeError(
                "router trace provenance must be TraceProvenance"
            )
        if not self.decisions:
            raise HBServeError("router trace must not be empty")
        prior: tuple[str, int, int] | None = None
        for decision in self.decisions:
            if not isinstance(decision, RouterDecision):
                raise HBServeError("router trace contains a non-decision")
            if prior is not None and decision.key <= prior:
                raise HBServeError(
                    "router decisions must be strictly sorted and unique"
                )
            prior = decision.key

    @cached_property
    def _by_key(self) -> Mapping[tuple[str, int, int], tuple[int, ...]]:
        return {decision.key: decision.experts for decision in self.decisions}

    def experts_for(
        self,
        *,
        request: RequestSpec,
        token_index: int,
        layer: int,
        model: ModelSpec,
    ) -> tuple[int, ...]:
        key = (request.request_id, token_index, layer)
        try:
            experts = self._by_key[key]
        except KeyError as error:
            raise HBServeError(
                "router trace is missing decision "
                f"request={request.request_id}, token={token_index}, layer={layer}"
            ) from error
        self._validate_one(model=model, layer=layer, experts=experts)
        return experts

    @staticmethod
    def _validate_one(
        *, model: ModelSpec, layer: int, experts: tuple[int, ...]
    ) -> None:
        if layer >= model.num_layers:
            raise HBServeError("router layer exceeds model layers")
        layer_spec = model.layers[layer]
        if not layer_spec.is_moe:
            raise HBServeError("router decision targets a dense layer")
        if len(experts) != layer_spec.top_k:
            raise HBServeError(
                "router decision width differs from model top_k"
            )
        if len(experts) != len(set(experts)) or any(
            expert >= len(layer_spec.expert_weight_bytes) for expert in experts
        ):
            raise HBServeError(
                "router decision has duplicate or out-of-range experts"
            )

    def validate_complete(
        self,
        *,
        requests: Sequence[RequestSpec],
        models: Mapping[str, ModelSpec],
    ) -> None:
        requests_by_id = {request.request_id: request for request in requests}
        if len(requests_by_id) != len(requests):
            raise HBServeError(
                "router validation received duplicate request IDs"
            )
        expected_count = 0
        for request in requests:
            try:
                model = models[request.model_id]
            except KeyError as error:
                raise HBServeError(
                    f"request names unknown model {request.model_id}"
                ) from error
            moe_layers = sum(layer.is_moe for layer in model.layers)
            expected_count = _checked_add(
                expected_count,
                _checked_mul(
                    request.processed_input_tokens,
                    moe_layers,
                    "router expected decisions",
                ),
                "router expected decisions",
            )
        invalid: list[tuple[str, int, int]] = []
        for decision in self.decisions:
            request = requests_by_id.get(decision.request_id)
            if request is None:
                invalid.append(decision.key)
                continue
            model = models[request.model_id]
            if (
                decision.token_index >= request.processed_input_tokens
                or decision.layer >= model.num_layers
                or not model.layers[decision.layer].is_moe
            ):
                invalid.append(decision.key)
                continue
            self._validate_one(
                model=model,
                layer=decision.layer,
                experts=decision.experts,
            )
        if len(self.decisions) != expected_count or invalid:
            valid_count = len(self.decisions) - len(invalid)
            missing_count = expected_count - valid_count
            extra_count = len(invalid)
            sample_missing: list[tuple[str, int, int]] = []
            if missing_count:
                actual = self._by_key
                for request in requests:
                    model = models[request.model_id]
                    for token in range(request.processed_input_tokens):
                        for layer, layer_spec in enumerate(model.layers):
                            key = (request.request_id, token, layer)
                            if layer_spec.is_moe and key not in actual:
                                sample_missing.append(key)
                                if len(sample_missing) == 3:
                                    break
                        if len(sample_missing) == 3:
                            break
                    if len(sample_missing) == 3:
                        break
            raise HBServeError(
                "router trace coverage differs from the request/model workload: "
                f"missing={sample_missing} ({missing_count} total), "
                f"invalid_or_extra={invalid[:3]} "
                f"({extra_count} total)"
            )

    def canonical(self) -> dict[str, Any]:
        return {
            "schema": ROUTER_TRACE_SCHEMA,
            "provenance": self.provenance.canonical(),
            "decisions": [decision.canonical() for decision in self.decisions],
        }

    @cached_property
    def digest(self) -> str:
        return canonical_sha256(self.canonical())


@dataclass(frozen=True)
class BatchSlice:
    """One request's contribution to one scheduler iteration.

    ``phase`` is per slice: a mixed iteration carries decode slices (one
    token each, always emitting) next to prefill chunks.  ``token_begin`` is
    the request's live context length when the slice starts; a prefill chunk
    may extend past the prompt only when it recomputes context that a
    preemption discarded.
    """

    request_id: str
    token_begin: int
    token_count: int
    context_tokens_before: int
    emits_output: bool
    phase: str

    def __post_init__(self) -> None:
        _entity_identifier(self.request_id, "batch-slice request id")
        _integer(self.token_begin, "batch-slice token_begin")
        _integer(self.token_count, "batch-slice token_count", minimum=1)
        _integer(
            self.context_tokens_before,
            "batch-slice context_tokens_before",
        )
        if not isinstance(self.emits_output, bool):
            raise HBServeError(
                "batch-slice emits_output must be boolean"
            )
        if self.phase not in SLICE_PHASES:
            raise HBServeError("batch-slice phase must be prefill or decode")
        if self.token_begin != self.context_tokens_before:
            raise HBServeError(
                "schema v2 requires token_begin == context_tokens_before"
            )
        if self.phase == "decode" and (
            self.token_count != 1 or not self.emits_output
        ):
            raise HBServeError(
                "decode slices process one token and emit output"
            )

    @property
    def token_end(self) -> int:
        return self.token_begin + self.token_count

    def canonical(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "token_begin": self.token_begin,
            "token_count": self.token_count,
            "context_tokens_before": self.context_tokens_before,
            "emits_output": self.emits_output,
            "phase": self.phase,
        }


@dataclass(frozen=True)
class ScheduledBatch:
    batch_id: int
    model_id: str
    slices: tuple[BatchSlice, ...]
    not_before_ns: float

    def __post_init__(self) -> None:
        _integer(self.batch_id, "scheduled batch id")
        _entity_identifier(self.model_id, "scheduled batch model id")
        if not self.slices:
            raise HBServeError("scheduled batch must contain requests")
        if not all(isinstance(item, BatchSlice) for item in self.slices):
            raise HBServeError(
                "scheduled batch slices must be BatchSlice values"
            )
        _finite(self.not_before_ns, "scheduled batch not_before_ns")
        identifiers = [item.request_id for item in self.slices]
        if len(identifiers) != len(set(identifiers)):
            raise HBServeError("scheduled batch repeats a request")

    @property
    def scheduled_tokens(self) -> int:
        return sum(item.token_count for item in self.slices)

    @property
    def emits_output(self) -> bool:
        return any(item.emits_output for item in self.slices)

    @property
    def kind(self) -> str:
        """``prefill``, ``decode``, or ``mixed`` (both phases in one pass)."""

        phases = {item.phase for item in self.slices}
        if len(phases) == 1:
            return next(iter(phases))
        return "mixed"

    def canonical(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "model_id": self.model_id,
            "kind": self.kind,
            "not_before_ns": self.not_before_ns,
            "slices": [item.canonical() for item in self.slices],
        }


@dataclass(frozen=True)
class SchedulerPolicy:
    """Token-budgeted mixed iterations (vLLM-V1 style).

    Every iteration admits all runnable decode requests plus prefill chunks of
    waiting requests until ``max_batch_tokens`` or ``max_batch_requests`` is
    reached; each prefill chunk is at most ``prefill_chunk_tokens``.
    """

    max_batch_requests: int
    max_batch_tokens: int
    prefill_chunk_tokens: int

    def __post_init__(self) -> None:
        _integer(self.max_batch_requests, "max_batch_requests", minimum=1)
        _integer(self.max_batch_tokens, "max_batch_tokens", minimum=1)
        _integer(self.prefill_chunk_tokens, "prefill_chunk_tokens", minimum=1)

    def canonical(self) -> dict[str, Any]:
        return {
            "max_batch_requests": self.max_batch_requests,
            "max_batch_tokens": self.max_batch_tokens,
            "prefill_chunk_tokens": self.prefill_chunk_tokens,
            "batch_model_policy": "one_model_per_batch_fifo_v1",
            "batch_phase_policy": "mixed_iteration_v1",
        }


@dataclass(frozen=True)
class BatchTiming:
    """Total per-layer compute, its pre-routing portion for MoE, and the tail."""

    layer_ns: tuple[float, ...]
    tail_ns: float
    timing_model: str
    evidence_state: str
    routing_ns: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        for index, value in enumerate(self.layer_ns):
            _finite(value, f"layer_ns[{index}]")
        _finite(self.tail_ns, "tail_ns")
        if self.routing_ns and len(self.routing_ns) != len(self.layer_ns):
            raise HBServeError("routing timing must cover every layer")
        for index, value in enumerate(self.routing_ns):
            _finite(value, f"routing_ns[{index}]")
            if value > self.layer_ns[index]:
                raise HBServeError("routing timing cannot exceed total layer timing")
        if self.timing_model not in TIMING_MODELS:
            raise HBServeError(
                f"unsupported timing model: {self.timing_model}"
            )
        if self.evidence_state not in TIMING_EVIDENCE_STATES:
            raise HBServeError(
                f"unsupported timing evidence state: {self.evidence_state}"
            )


class TimingProvider(Protocol):
    timing_model: str
    evidence_state: str

    @property
    def includes_compute(self) -> bool: ...

    def canonical(self) -> dict[str, Any]: ...

    def timing_for(
        self,
        *,
        model: ModelSpec,
        batch: ScheduledBatch,
    ) -> BatchTiming: ...


@dataclass(frozen=True)
class MemoryOnlyTimingProvider:
    """Zero compute: batch time is the memory critical path alone.

    Results from this provider are memory critical paths, never TTFT/TPOT.
    """

    timing_model: str = "memory_only"
    evidence_state: str = "memory_only"

    @property
    def includes_compute(self) -> bool:
        return False

    def canonical(self) -> dict[str, Any]:
        return {"type": self.timing_model}

    def timing_for(
        self,
        *,
        model: ModelSpec,
        batch: ScheduledBatch,
    ) -> BatchTiming:
        return BatchTiming(
            layer_ns=(0.0,) * model.num_layers,
            tail_ns=0.0,
            timing_model=self.timing_model,
            evidence_state=self.evidence_state,
        )


@dataclass(frozen=True)
class LinearTimingProvider:
    """Explicit per-layer sensitivity timing, not a hardware calibration.

    Every layer costs ``fixed_ns_per_layer + ns_per_token_per_layer x
    scheduled tokens``; the tail costs ``tail_fixed_ns +
    tail_ns_per_output_request x emitting requests``.  The model ignores
    context length, so it cannot represent attention; use ``roofline`` for
    that.
    """

    fixed_ns_per_layer: float
    ns_per_token_per_layer: float
    tail_fixed_ns: float
    tail_ns_per_output_request: float
    timing_model: str = "linear"
    evidence_state: str = "modeled_sensitivity"
    moe_routing_fraction: float | None = None

    def __post_init__(self) -> None:
        if self.moe_routing_fraction is not None:
            if _finite(self.moe_routing_fraction, "moe_routing_fraction") > 1.0:
                raise HBServeError("moe_routing_fraction must be in [0, 1]")
        for name, value in (
            ("fixed_ns_per_layer", self.fixed_ns_per_layer),
            ("ns_per_token_per_layer", self.ns_per_token_per_layer),
            ("tail_fixed_ns", self.tail_fixed_ns),
            ("tail_ns_per_output_request", self.tail_ns_per_output_request),
        ):
            _finite(value, name)
        if self.timing_model != "linear":
            raise HBServeError("linear timing provider has a fixed model label")
        if self.evidence_state != "modeled_sensitivity":
            raise HBServeError(
                "linear timing is sensitivity-only; a calibrated label requires "
                "a digest-bound calibration provider"
            )

    @property
    def includes_compute(self) -> bool:
        return True

    def canonical(self) -> dict[str, Any]:
        return {
            "type": self.timing_model,
            "fixed_ns_per_layer": self.fixed_ns_per_layer,
            "ns_per_token_per_layer": self.ns_per_token_per_layer,
            "tail_fixed_ns": self.tail_fixed_ns,
            "tail_ns_per_output_request": self.tail_ns_per_output_request,
            **({"moe_routing_fraction": self.moe_routing_fraction}
               if self.moe_routing_fraction is not None else {}),
        }

    def timing_for(
        self,
        *,
        model: ModelSpec,
        batch: ScheduledBatch,
    ) -> BatchTiming:
        tokens = batch.scheduled_tokens
        per_layer = self.fixed_ns_per_layer + self.ns_per_token_per_layer * tokens
        if (
            per_layer
            and any(layer.is_moe for layer in model.layers)
            and self.moe_routing_fraction is None
        ):
            raise HBServeError(
                "linear MoE timing requires explicit moe_routing_fraction"
            )
        output_requests = sum(item.emits_output for item in batch.slices)
        tail = (
            self.tail_fixed_ns
            + self.tail_ns_per_output_request * output_requests
            if output_requests
            else 0.0
        )
        return BatchTiming(
            layer_ns=(per_layer,) * model.num_layers,
            routing_ns=tuple(
                per_layer * (self.moe_routing_fraction or 0.0)
                if layer.is_moe else 0.0
                for layer in model.layers
            ),
            tail_ns=tail,
            timing_model=self.timing_model,
            evidence_state=self.evidence_state,
        )


@dataclass(frozen=True)
class RooflineTimingProvider:
    """Per-layer compute time from the model's FLOP ledger.

    ``layer_ns = FLOPs_layer / (peak_tflops x 1e12 x efficiency) x 1e9`` where
    ``FLOPs_layer`` sums, over the iteration's slices, ``tokens x
    flops_per_token`` plus the causal attention term
    ``attention_flops_per_context_token x sum_{i<tokens}(context_before + i
    + 1)``.  The tail costs ``lm_head_flops_per_token`` per emitting request.
    Memory time is not added here: the compiler's DAG lets the next layer's
    weight and KV traffic prefetch during this compute barrier, so one
    iteration takes approximately ``max(memory, compute)`` per layer.
    """

    peak_tflops: float
    efficiency: float
    timing_model: str = "roofline"
    evidence_state: str = "modeled_roofline"

    def __post_init__(self) -> None:
        if _finite(self.peak_tflops, "peak_tflops") <= 0.0:
            raise HBServeError("peak_tflops must be > 0")
        efficiency = _finite(self.efficiency, "efficiency")
        if not 0.0 < efficiency <= 1.0:
            raise HBServeError("efficiency must be in (0, 1]")
        if self.timing_model != "roofline":
            raise HBServeError("roofline timing provider has a fixed model label")
        if self.evidence_state != "modeled_roofline":
            raise HBServeError(
                "roofline timing provider has a fixed evidence state"
            )

    @property
    def includes_compute(self) -> bool:
        return True

    @property
    def flops_per_ns(self) -> float:
        return self.peak_tflops * 1e12 * self.efficiency / 1e9

    def canonical(self) -> dict[str, Any]:
        return {
            "type": self.timing_model,
            "peak_tflops": self.peak_tflops,
            "efficiency": self.efficiency,
        }

    def layer_flops(self, layer: LayerSpec, batch: ScheduledBatch) -> int:
        total = 0
        for item in batch.slices:
            tokens = item.token_count
            context = item.context_tokens_before
            causal_context = tokens * context + tokens * (tokens + 1) // 2
            total += (
                tokens * layer.flops_per_token
                + causal_context * layer.attention_flops_per_context_token
            )
        return total

    def timing_for(
        self,
        *,
        model: ModelSpec,
        batch: ScheduledBatch,
    ) -> BatchTiming:
        rate = self.flops_per_ns
        routing_ns: list[float] = []
        for layer in model.layers:
            if not layer.is_moe:
                routing_ns.append(0.0)
                continue
            if layer.pre_routing_flops_per_token is None:
                raise HBServeError(
                    "roofline MoE timing requires pre_routing_flops_per_token"
                )
            routing_flops = sum(
                item.token_count * layer.pre_routing_flops_per_token
                + (item.token_count * item.context_tokens_before
                   + item.token_count * (item.token_count + 1) // 2)
                * layer.attention_flops_per_context_token
                for item in batch.slices
            )
            routing_ns.append(routing_flops / rate)
        layer_ns = tuple(
            self.layer_flops(layer, batch) / rate for layer in model.layers
        )
        output_requests = sum(item.emits_output for item in batch.slices)
        tail = output_requests * model.lm_head_flops_per_token / rate
        return BatchTiming(
            layer_ns=layer_ns,
            routing_ns=tuple(routing_ns),
            tail_ns=tail,
            timing_model=self.timing_model,
            evidence_state=self.evidence_state,
        )


@dataclass(frozen=True)
class SemanticOperation:
    id: str
    op: str | None
    object_id: str | None
    offset: int
    bytes: int
    duration_ns: float
    dependencies: tuple[str, ...]
    role: str

    @property
    def is_barrier(self) -> bool:
        return self.op is None

    def __post_init__(self) -> None:
        _identifier(self.id, "semantic operation id")
        _identifier(self.role, f"semantic operation {self.id} role")
        _finite(self.duration_ns, f"semantic operation {self.id} duration_ns")
        if self.is_barrier:
            if self.object_id is not None or self.offset != 0 or self.bytes != 0:
                raise HBServeError(
                    f"semantic barrier {self.id} has memory fields"
                )
        else:
            if self.op not in {"R", "W"} or self.object_id is None:
                raise HBServeError(
                    f"semantic memory operation {self.id} is malformed"
                )
            _identifier(self.object_id, f"semantic operation {self.id} object")
            _integer(self.offset, f"semantic operation {self.id} offset")
            _integer(self.bytes, f"semantic operation {self.id} bytes", minimum=1)
            if self.duration_ns != 0.0:
                raise HBServeError(
                    f"semantic memory operation {self.id} has a duration"
                )
            _checked_add(self.offset, self.bytes, f"semantic operation {self.id}")
        if len(self.dependencies) != len(set(self.dependencies)):
            raise HBServeError(
                f"semantic operation {self.id} repeats a dependency"
            )
        for dependency in self.dependencies:
            _identifier(dependency, f"semantic operation {self.id} dependency")
            if dependency == self.id:
                raise HBServeError(
                    f"semantic operation {self.id} depends on itself"
                )

    def canonical(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "record": "barrier" if self.is_barrier else "memory",
            "op": self.op,
            "object_id": self.object_id,
            "offset": self.offset,
            "bytes": self.bytes,
            "duration_ns": self.duration_ns,
            "dependencies": list(self.dependencies),
            "role": self.role,
        }


@dataclass(frozen=True)
class CanonicalServingBatch:
    schedule: ScheduledBatch
    model_sha256: str
    request_trace_sha256: str
    router_trace_sha256: str | None
    timing_model: str
    timing_evidence_state: str
    prefetch_depth: int
    operations: tuple[SemanticOperation, ...]
    audit: Mapping[str, Mapping[str, Any]]

    def __post_init__(self) -> None:
        if self.timing_model not in TIMING_MODELS:
            raise HBServeError("canonical batch timing model is unsupported")
        if self.timing_evidence_state not in TIMING_EVIDENCE_STATES:
            raise HBServeError(
                "canonical batch timing evidence state is unsupported"
            )
        _integer(self.prefetch_depth, "canonical batch prefetch depth")
        for name, digest in (
            ("model", self.model_sha256),
            ("request trace", self.request_trace_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise HBServeError(f"{name} digest is malformed")
        _sha256_or_none(self.router_trace_sha256, "router trace")
        if not self.operations:
            raise HBServeError("canonical serving batch is empty")
        seen: set[str] = set()
        for operation in self.operations:
            if operation.id in seen:
                raise HBServeError(
                    f"duplicate semantic operation ID: {operation.id}"
                )
            for dependency in operation.dependencies:
                if dependency not in seen:
                    raise HBServeError(
                        f"semantic operation {operation.id} has a forward or "
                        f"missing dependency {dependency}"
                    )
            seen.add(operation.id)
        if set(self.audit) != seen:
            raise HBServeError(
                "canonical serving audit must cover every operation exactly"
            )
        if any(not isinstance(value, Mapping) for value in self.audit.values()):
            raise HBServeError("canonical serving audit rows must be objects")
        for operation in self.operations:
            if self.audit[operation.id].get("role") != operation.role:
                raise HBServeError(
                    f"canonical audit role differs for {operation.id}"
                )

    @cached_property
    def memory_operations(self) -> tuple[SemanticOperation, ...]:
        return tuple(item for item in self.operations if not item.is_barrier)

    @cached_property
    def logical_bytes(self) -> int:
        total = 0
        for operation in self.memory_operations:
            total = _checked_add(total, operation.bytes, "canonical logical bytes")
        return total

    def canonical(self) -> dict[str, Any]:
        return {
            "schema": CANONICAL_BATCH_SCHEMA,
            "schedule": self.schedule.canonical(),
            "model_sha256": self.model_sha256,
            "request_trace_sha256": self.request_trace_sha256,
            "router_trace_sha256": self.router_trace_sha256,
            "timing_model": self.timing_model,
            "timing_evidence_state": self.timing_evidence_state,
            "prefetch_depth": self.prefetch_depth,
            "operations": [operation.canonical() for operation in self.operations],
            "audit": {
                operation_id: dict(self.audit[operation_id])
                for operation_id in sorted(self.audit)
            },
        }

    def audit_summary(self) -> dict[str, Any]:
        roles: dict[str, dict[str, int]] = {}
        token_id_sources: dict[str, int] = {}
        expert_routes: list[dict[str, int]] = []
        barrier_duration_ns = 0.0
        for operation in self.operations:
            row = roles.setdefault(
                operation.role,
                {
                    "operations": 0,
                    "read_bytes": 0,
                    "write_bytes": 0,
                    "barriers": 0,
                },
            )
            if operation.is_barrier:
                row["barriers"] += 1
                barrier_duration_ns += operation.duration_ns
                continue
            row["operations"] += 1
            row["read_bytes" if operation.op == "R" else "write_bytes"] += (
                operation.bytes
            )
            labels = self.audit[operation.id]
            token_source = labels.get("token_id_source")
            if token_source is not None:
                source = str(token_source)
                token_id_sources[source] = token_id_sources.get(source, 0) + 1
            if operation.role == "moe/routed_expert_weights":
                expert_routes.append(
                    {
                        "layer": int(labels["layer"]),
                        "expert": int(labels["expert"]),
                        "routed_tokens": int(labels["routed_tokens"]),
                    }
                )
        return {
            "logical_bytes": self.logical_bytes,
            "memory_operations": len(self.memory_operations),
            "barriers": len(self.operations) - len(self.memory_operations),
            "modeled_compute_duration_ns": barrier_duration_ns,
            "roles": {role: roles[role] for role in sorted(roles)},
            "token_id_sources": dict(sorted(token_id_sources.items())),
            "moe_expert_routes": sorted(
                expert_routes,
                key=lambda row: (row["layer"], row["expert"]),
            ),
        }

    @cached_property
    def digest(self) -> str:
        return canonical_sha256(self.canonical())
