"""Dense Hugging Face geometry and actual native batches for HBServe."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from hbserve.compiler import HBServeCompiler
from hbserve.contracts import (
    BatchSlice, LayerSpec, ModelSpec, RequestSpec, RequestTrace,
    RooflineTimingProvider, ScheduledBatch, TraceProvenance,
)

DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4,
               "fp8_e4m3": 1}
DTYPE_NAMES = {"float16": "FP16", "bfloat16": "BF16", "float32": "FP32",
               "fp8_e4m3": "FP8"}


def positive_int(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def dense_model(hf: dict, dtype: str, kv_dtype: str) -> ModelSpec:
    """Derive a dense Llama/Qwen/Mistral ledger, including norms and biases."""
    kind = hf.get("model_type")
    if kind not in {"llama", "qwen2", "qwen3", "mistral"}:
        raise ValueError(f"unsupported native model {kind!r}; use dense Llama/Qwen/Mistral")
    if any(hf.get(k) for k in ("num_experts", "n_routed_experts", "num_local_experts",
                               "quantization_config", "kv_lora_rank")):
        raise ValueError("MoE, quantized weights and compressed KV need explicit traffic models")
    if hf.get("use_sliding_window") or (hf.get("sliding_window") and hf.get("use_sliding_window", True)):
        raise ValueError("windowed attention is not the dense full-context traffic model")
    if dtype not in {"float16", "bfloat16", "float32"} or kv_dtype not in DTYPE_BYTES:
        raise ValueError("unsupported weight or KV dtype")
    h = positive_int(hf.get("hidden_size"), "hidden_size")
    f = positive_int(hf.get("intermediate_size"), "intermediate_size")
    n = positive_int(hf.get("num_hidden_layers"), "num_hidden_layers")
    q = positive_int(hf.get("num_attention_heads"), "num_attention_heads")
    k = positive_int(hf.get("num_key_value_heads", q), "num_key_value_heads")
    d = positive_int(hf.get("head_dim", h // q), "head_dim")
    v = positive_int(hf.get("vocab_size"), "vocab_size")
    if q % k:
        raise ValueError("query heads must be divisible by KV heads")
    b, kb = DTYPE_BYTES[dtype], DTYPE_BYTES[kv_dtype]
    attention_matrix = h * (q + 2 * k) * d + q * d * h
    ffn_matrix = 3 * h * f
    # Qwen2 has Q/K/V bias by default; Qwen3 additionally has Q/K RMSNorm.
    qkv_bias = bool(hf.get("attention_bias", kind == "qwen2"))
    out_bias = bool(hf.get("attention_bias", False))
    attention_parameters = attention_matrix + h + (2 * d if kind == "qwen3" else 0)
    attention_parameters += ((q + 2 * k) * d if qkv_bias else 0) + (h if out_bias else 0)
    ffn_parameters = ffn_matrix + h + ((2 * f + h) if hf.get("mlp_bias", False) else 0)
    layer = LayerSpec(
        attention_weight_bytes=attention_parameters * b,
        ffn_weight_bytes=ffn_parameters * b,
        router_weight_bytes=0, shared_expert_weight_bytes=0,
        expert_weight_bytes=(), top_k=0, kv_bytes_per_token=2 * k * d * kb,
        flops_per_token=2 * (attention_matrix + ffn_matrix),
        attention_flops_per_context_token=4 * q * d,
    )
    return ModelSpec(
        model_id="native", provenance={"kind": "checkpoint_manifest",
            "source": "Hugging Face config; derived dense geometry, modeled compute",
            "sha256": hashlib.sha256(json.dumps(hf, sort_keys=True).encode()).hexdigest()},
        vocab_size=v, embedding_bytes=v * h * b, final_norm_bytes=h * b,
        lm_head_bytes=v * h * b, tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
        layers=(layer,) * n, lm_head_flops_per_token=2 * v * h,
    )


@dataclass(frozen=True)
class NativeRequest:
    rid: str
    slots: tuple[int, ...]
    tokens: tuple[int, ...]
    prompt_tokens: int
    output_tokens: int
    phase: str
    emits_output: bool

    @property
    def key(self) -> str:
        return hashlib.sha256(self.rid.encode()).hexdigest()

    @property
    def past(self) -> int:
        return len(self.slots) - len(self.tokens)

    def validate(self, pool: int, vocab: int) -> None:
        if not self.rid or not self.tokens or self.past < 0:
            raise ValueError("native batch needs identity, input tokens and complete KV positions")
        if len(set(self.slots)) != len(self.slots) or any(type(s) is not int or not 0 < s < pool for s in self.slots):
            raise ValueError("native KV slots alias, touch reserved slot zero, or exceed capacity")
        if any(type(t) is not int or not 0 <= t < vocab for t in self.tokens):
            raise ValueError("native input token is outside the vocabulary")
        positive_int(self.prompt_tokens, "prompt_tokens")
        positive_int(self.output_tokens, "output_tokens")
        if len(self.slots) > self.prompt_tokens + self.output_tokens - 1:
            raise ValueError("native batch exceeds request input-token lifetime")
        if self.phase not in {"prefill", "decode"}:
            raise ValueError("unsupported native forward mode")


class NativeCompiler(HBServeCompiler):
    """Use the existing object DAG with pre-forward token IDs, never surrogates."""

    def __init__(self, model: ModelSpec, rows: list[NativeRequest], timing: RooflineTimingProvider):
        self.native_rows = {r.key: r for r in rows}
        requests = tuple(sorted((RequestSpec(r.key, 0, model.model_id,
            r.prompt_tokens, r.output_tokens) for r in rows), key=lambda r: r.request_id))
        super().__init__(models={model.model_id: model}, request_trace=RequestTrace(
            TraceProvenance("synthetic_sensitivity", "native SGLang batch snapshot"), requests),
            timing=timing)

    def _token_id(self, model, request, token_index):
        row = self.native_rows[request.request_id]
        return row.tokens[token_index - row.past], "native_sglang_pre_forward"

    def batch(self, batch_id: int, now_ns: float):
        return self.compile(ScheduledBatch(batch_id, "native", tuple(
            BatchSlice(r.key, r.past, len(r.tokens), r.past, r.emits_output, r.phase)
            for r in self.native_rows.values()), now_ns))
