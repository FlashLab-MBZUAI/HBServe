#!/usr/bin/env python3
"""Strict JSON input/output for HBServe workloads."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from hbserve.contracts import (
    REQUEST_TRACE_SCHEMA,
    ROUTER_TRACE_SCHEMA,
    LinearTimingProvider,
    MemoryOnlyTimingProvider,
    ModelSpec,
    RequestSpec,
    RequestTrace,
    RooflineTimingProvider,
    RouterDecision,
    RouterTrace,
    SchedulerPolicy,
    HBServeError,
    TimingProvider,
    TraceProvenance,
)
from hbserve.placement import PLACEMENT_SCHEMA, KvPlacement, PlacementSpec
from hbserve.synthetic import (
    HotsetZipfRouter,
    SyntheticRequestConfig,
)


SYNTHETIC_REQUEST_SCHEMA = {
    "name": "hbserve.synthetic_requests",
    "version": 1,
}
SYNTHETIC_ROUTER_SCHEMA = {
    "name": "hbserve.synthetic_router",
    "version": 1,
}
RUN_CONFIG_SCHEMA = {
    "name": "hbserve.run_config",
    "version": 2,
}
DEFAULT_PREFETCH_DEPTH = 1


def _pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HBServeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_object(path: Path, description: str) -> dict[str, Any]:
    if path.is_symlink():
        raise HBServeError(
            f"{description} must be a regular non-symlink file: {path}"
        )
    resolved = path.resolve()
    if not resolved.is_file():
        raise HBServeError(
            f"{description} must be a regular non-symlink file: {resolved}"
        )
    try:
        value = json.loads(
            resolved.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                HBServeError(f"non-finite JSON number: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HBServeError(
            f"cannot parse {description} {resolved}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise HBServeError(f"{description} must be a JSON object")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], description: str) -> None:
    if set(value) != keys:
        raise HBServeError(
            f"{description} fields differ: "
            f"missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)}"
        )


def _mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HBServeError(f"{description} must be an object")
    return dict(value)


def _integer(value: Any, description: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise HBServeError(
            f"{description} must be an integer >= {minimum}"
        )
    return value


def _string(value: Any, description: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        suffix = " non-empty" if nonempty else ""
        raise HBServeError(f"{description} must be a{suffix} string")
    return value


def _number(value: Any, description: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HBServeError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise HBServeError(
            f"{description} must be finite and >= {minimum}"
        )
    return result


def _provenance(value: Any, description: str) -> TraceProvenance:
    raw = _mapping(value, description)
    _exact(raw, {"kind", "source", "sha256"}, description)
    sha = raw["sha256"]
    if sha is not None and not isinstance(sha, str):
        raise HBServeError(f"{description}.sha256 must be null or a string")
    return TraceProvenance(
        kind=_string(raw["kind"], f"{description}.kind"),
        source=_string(raw["source"], f"{description}.source"),
        sha256=sha,
    )


def load_models(paths: Sequence[Path]) -> dict[str, ModelSpec]:
    if not paths:
        raise HBServeError("at least one serving model is required")
    result: dict[str, ModelSpec] = {}
    for path in paths:
        model = ModelSpec.from_dict(load_json_object(path, "serving model"))
        if model.model_id in result:
            raise HBServeError(
                f"duplicate serving model ID: {model.model_id}"
            )
        result[model.model_id] = model
    return result


def load_request_trace(path: Path) -> RequestTrace:
    value = load_json_object(path, "request trace")
    _exact(value, {"schema", "provenance", "requests"}, "request trace")
    if value["schema"] != REQUEST_TRACE_SCHEMA:
        raise HBServeError("unsupported request trace schema")
    raw_requests = value["requests"]
    if not isinstance(raw_requests, list):
        raise HBServeError("request trace requests must be an array")
    requests: list[RequestSpec] = []
    for index, raw_value in enumerate(raw_requests):
        raw = {"cache_salt": "", **_mapping(raw_value, f"requests[{index}]")}
        _exact(
            raw,
            {
                "request_id",
                "arrival_ns",
                "model_id",
                "prompt_tokens",
                "output_tokens",
                "token_ids",
                "cache_salt",
            },
            f"requests[{index}]",
        )
        raw_tokens = raw["token_ids"]
        if raw_tokens is not None and not isinstance(raw_tokens, list):
            raise HBServeError(
                f"requests[{index}].token_ids must be null or an array"
            )
        requests.append(
            RequestSpec(
                request_id=_string(raw["request_id"], "request ID"),
                cache_salt=raw["cache_salt"],
                arrival_ns=_number(raw["arrival_ns"], "request arrival"),
                model_id=_string(raw["model_id"], "request model ID"),
                prompt_tokens=_integer(
                    raw["prompt_tokens"], "request prompt_tokens", minimum=1
                ),
                output_tokens=_integer(
                    raw["output_tokens"], "request output_tokens", minimum=1
                ),
                token_ids=(
                    None
                    if raw_tokens is None
                    else tuple(
                        _integer(token, "request token ID")
                        for token in raw_tokens
                    )
                ),
            )
        )
    return RequestTrace(
        provenance=_provenance(value["provenance"], "request provenance"),
        requests=tuple(requests),
    )


def load_router(path: Path) -> RouterTrace | HotsetZipfRouter:
    value = load_json_object(path, "router workload")
    schema = value.get("schema")
    if schema == ROUTER_TRACE_SCHEMA:
        _exact(value, {"schema", "provenance", "decisions"}, "router trace")
        raw_decisions = value["decisions"]
        if not isinstance(raw_decisions, list):
            raise HBServeError("router decisions must be an array")
        decisions: list[RouterDecision] = []
        for index, raw_value in enumerate(raw_decisions):
            raw = _mapping(raw_value, f"router decisions[{index}]")
            _exact(
                raw,
                {"request_id", "token_index", "layer", "experts"},
                f"router decisions[{index}]",
            )
            experts = raw["experts"]
            if not isinstance(experts, list):
                raise HBServeError("router experts must be an array")
            decisions.append(
                RouterDecision(
                    request_id=_string(raw["request_id"], "router request ID"),
                    token_index=_integer(raw["token_index"], "router token"),
                    layer=_integer(raw["layer"], "router layer"),
                    experts=tuple(
                        _integer(expert, "router expert") for expert in experts
                    ),
                )
            )
        return RouterTrace(
            provenance=_provenance(
                value["provenance"], "router trace provenance"
            ),
            decisions=tuple(decisions),
        )
    if schema == SYNTHETIC_ROUTER_SCHEMA:
        _exact(
            value,
            {"schema", "seed", "hot_experts", "hot_mass", "alpha"},
            "synthetic router",
        )
        return HotsetZipfRouter(
            seed=_integer(value["seed"], "router seed"),
            hot_experts=_integer(
                value["hot_experts"], "router hot_experts", minimum=1
            ),
            hot_mass=_number(value["hot_mass"], "router hot_mass"),
            alpha=_number(value["alpha"], "router alpha"),
        )
    raise HBServeError("unsupported router workload schema")


PLACEMENT_FIELDS = {
    "schema",
    "hbm_capacity_bytes",
    "hbf_capacity_bytes",
    "external_capacity_bytes",
    "hbm_runtime_reserve_bytes",
    "hbm_model_cache_bytes",
    "model_weight_tiers",
    "object_tier_overrides",
    "model_load_chunk_bytes",
    "initial_cached_models",
    "hbm_alignment_bytes",
    "hbf_page_size_bytes",
    "external_page_size_bytes",
    "kv_block_tokens",
    "kv_placement",
    "prefix_cache_bytes",
    "prefix_cache_ttl_ns",
}


def placement_from_dict(value: Mapping[str, Any]) -> PlacementSpec:
    value = {"prefix_cache_bytes": 0, "prefix_cache_ttl_ns": None, **value}
    _exact(value, PLACEMENT_FIELDS, "serving placement")
    if value["schema"] != PLACEMENT_SCHEMA:
        raise HBServeError(
            "unsupported serving placement schema; expected "
            f"{PLACEMENT_SCHEMA['name']} v{PLACEMENT_SCHEMA['version']}"
        )
    tiers = _mapping(value["model_weight_tiers"], "model_weight_tiers")
    overrides = _mapping(value["object_tier_overrides"], "object overrides")
    cached = value["initial_cached_models"]
    if not isinstance(cached, list):
        raise HBServeError("initial_cached_models must be an array")
    kv_placement = _mapping(value["kv_placement"], "kv_placement")
    _exact(kv_placement, {"hot", "cold"}, "kv_placement")
    cold = kv_placement["cold"]
    if cold is not None:
        cold = _string(cold, "kv_placement.cold")
    return PlacementSpec(
        prefix_cache_bytes=_integer(value["prefix_cache_bytes"], "prefix cache bytes"),
        prefix_cache_ttl_ns=value["prefix_cache_ttl_ns"],
        hbm_capacity_bytes=_integer(
            value["hbm_capacity_bytes"], "HBM capacity", minimum=1
        ),
        hbf_capacity_bytes=_integer(value["hbf_capacity_bytes"], "HBF capacity"),
        external_capacity_bytes=_integer(
            value["external_capacity_bytes"], "external capacity"
        ),
        hbm_runtime_reserve_bytes=_integer(
            value["hbm_runtime_reserve_bytes"], "HBM runtime reserve"
        ),
        hbm_model_cache_bytes=_integer(
            value["hbm_model_cache_bytes"], "HBM model cache"
        ),
        model_weight_tiers={
            key: _string(item, f"model_weight_tiers.{key}")
            for key, item in tiers.items()
        },
        object_tier_overrides={
            key: _string(item, f"object_tier_overrides.{key}")
            for key, item in overrides.items()
        },
        model_load_chunk_bytes=_integer(
            value["model_load_chunk_bytes"],
            "model load chunk bytes",
            minimum=1,
        ),
        initial_cached_models=tuple(
            _string(item, "initial cached model") for item in cached
        ),
        hbm_alignment_bytes=_integer(
            value["hbm_alignment_bytes"], "HBM alignment", minimum=1
        ),
        hbf_page_size_bytes=_integer(
            value["hbf_page_size_bytes"], "HBF page size", minimum=1
        ),
        external_page_size_bytes=_integer(
            value["external_page_size_bytes"],
            "external page size",
            minimum=1,
        ),
        kv_block_tokens=_integer(
            value["kv_block_tokens"], "KV block tokens", minimum=1
        ),
        kv_placement=KvPlacement(
            hot=_string(kv_placement["hot"], "kv_placement.hot"),
            cold=cold,
        ),
    )


def load_placement(path: Path) -> PlacementSpec:
    return placement_from_dict(load_json_object(path, "serving placement"))


def timing_from_dict(
    timing: Mapping[str, Any], compute: Mapping[str, Any] | None
) -> tuple[TimingProvider, int]:
    """Build the timing provider and prefetch depth from run-config sections."""

    timing = _mapping(timing, "timing")
    timing_type = timing.get("type")
    prefetch_depth = _integer(
        timing.get("prefetch_depth", 0 if timing_type == "gpu_calibrated" else DEFAULT_PREFETCH_DEPTH),
        "timing prefetch_depth",
    )
    if timing_type == "roofline":
        _exact(timing, {"type", "prefetch_depth"}, "roofline timing")
        if compute is None:
            raise HBServeError("roofline timing requires a compute section")
        compute = _mapping(compute, "compute")
        _exact(compute, {"peak_tflops", "efficiency"}, "compute")
        return (
            RooflineTimingProvider(
                peak_tflops=_number(compute["peak_tflops"], "compute peak_tflops"),
                efficiency=_number(compute["efficiency"], "compute efficiency"),
            ),
            prefetch_depth,
        )
    if compute is not None:
        raise HBServeError(
            f"{timing_type} timing does not consume a compute section; set "
            "compute to null"
        )
    if timing_type == "gpu_calibrated":
        from hbserve.gpu_profile import GPUCalibratedTimingProvider
        _exact(timing, {"type", "profile", "model_bindings", "prefetch_depth"}, "GPU calibration timing")
        if prefetch_depth != 0:
            raise HBServeError("GPU calibration has no measured cross-layer prefetch")
        return GPUCalibratedTimingProvider(timing["profile"], timing["model_bindings"]), 0
    if timing_type == "memory_only":
        _exact(timing, {"type", "prefetch_depth"}, "memory-only timing")
        return MemoryOnlyTimingProvider(), prefetch_depth
    if timing_type == "linear":
        _exact(
            timing,
            {
                "type",
                "prefetch_depth",
                "fixed_ns_per_layer",
                "ns_per_token_per_layer",
                "tail_fixed_ns",
                "tail_ns_per_output_request",
            } | ({"moe_routing_fraction"} if "moe_routing_fraction" in timing else set()),
            "linear timing",
        )
        return (
            LinearTimingProvider(
                fixed_ns_per_layer=_number(
                    timing["fixed_ns_per_layer"], "linear fixed_ns_per_layer"
                ),
                ns_per_token_per_layer=_number(
                    timing["ns_per_token_per_layer"],
                    "linear ns_per_token_per_layer",
                ),
                tail_fixed_ns=_number(timing["tail_fixed_ns"], "linear tail_fixed_ns"),
                tail_ns_per_output_request=_number(
                    timing["tail_ns_per_output_request"],
                    "linear tail_ns_per_output_request",
                ),
                moe_routing_fraction=(
                    _number(timing["moe_routing_fraction"], "moe_routing_fraction")
                    if "moe_routing_fraction" in timing else None
                ),
            ),
            prefetch_depth,
        )
    raise HBServeError("timing type must be gpu_calibrated, roofline, memory_only, or linear")


def run_config_from_dict(
    value: Mapping[str, Any],
) -> tuple[SchedulerPolicy, TimingProvider, int]:
    _exact(value, {"schema", "scheduler", "timing", "compute"}, "serving run config")
    if value["schema"] != RUN_CONFIG_SCHEMA:
        raise HBServeError(
            "unsupported serving run config schema; expected "
            f"{RUN_CONFIG_SCHEMA['name']} v{RUN_CONFIG_SCHEMA['version']}"
        )
    scheduler = _mapping(value["scheduler"], "scheduler")
    _exact(
        scheduler,
        {"max_batch_requests", "max_batch_tokens", "prefill_chunk_tokens"},
        "scheduler",
    )
    policy = SchedulerPolicy(
        max_batch_requests=_integer(
            scheduler["max_batch_requests"], "max_batch_requests", minimum=1
        ),
        max_batch_tokens=_integer(
            scheduler["max_batch_tokens"], "max_batch_tokens", minimum=1
        ),
        prefill_chunk_tokens=_integer(
            scheduler["prefill_chunk_tokens"],
            "prefill_chunk_tokens",
            minimum=1,
        ),
    )
    compute = value["compute"]
    if compute is not None and not isinstance(compute, Mapping):
        raise HBServeError("compute must be an object or null")
    timing, prefetch_depth = timing_from_dict(value["timing"], compute)
    return policy, timing, prefetch_depth


def load_run_config(
    path: Path,
) -> tuple[SchedulerPolicy, TimingProvider, int]:
    return run_config_from_dict(load_json_object(path, "serving run config"))


def load_synthetic_request_config(path: Path) -> SyntheticRequestConfig:
    return synthetic_request_config_from_dict(
        load_json_object(path, "synthetic request config")
    )


def synthetic_request_config_from_dict(
    value: Mapping[str, Any],
) -> SyntheticRequestConfig:
    value = {"shared_prefix_tokens": 0, "shared_prefix_groups": 1, "prefix_reuse_probability": 1.0, **value}
    _exact(
        value,
        {
            "schema",
            "request_count",
            "arrival_rate_per_second",
            "prompt_lognormal_mean_tokens",
            "prompt_lognormal_sigma",
            "output_lognormal_mean_tokens",
            "output_lognormal_sigma",
            "model_probabilities",
            "seed",
            "first_arrival_policy",
            "shared_prefix_tokens",
            "shared_prefix_groups",
            "prefix_reuse_probability",
        },
        "synthetic request config",
    )
    if value["schema"] != SYNTHETIC_REQUEST_SCHEMA:
        raise HBServeError("unsupported synthetic request schema")
    probabilities = _mapping(
        value["model_probabilities"], "synthetic model probabilities"
    )
    return SyntheticRequestConfig(
        shared_prefix_tokens=_integer(value["shared_prefix_tokens"], "shared prefix tokens"),
        shared_prefix_groups=_integer(value["shared_prefix_groups"], "shared prefix groups", minimum=1),
        prefix_reuse_probability=value["prefix_reuse_probability"],
        request_count=_integer(
            value["request_count"], "request_count", minimum=1
        ),
        arrival_rate_per_second=_number(
            value["arrival_rate_per_second"],
            "arrival_rate_per_second",
            minimum=0.0,
        ),
        prompt_lognormal_mean_tokens=_number(
            value["prompt_lognormal_mean_tokens"],
            "prompt mean",
            minimum=0.0,
        ),
        prompt_lognormal_sigma=_number(
            value["prompt_lognormal_sigma"], "prompt sigma"
        ),
        output_lognormal_mean_tokens=_number(
            value["output_lognormal_mean_tokens"],
            "output mean",
            minimum=0.0,
        ),
        output_lognormal_sigma=_number(
            value["output_lognormal_sigma"], "output sigma"
        ),
        model_probabilities={
            str(key): _number(item, f"probability {key}", minimum=0.0)
            for key, item in probabilities.items()
        },
        seed=_integer(value["seed"], "synthetic seed"),
        first_arrival_policy=_string(
            value["first_arrival_policy"], "first_arrival_policy"
        ),
    )


def create_run_directory(root: Path) -> Path:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(tempfile.mkdtemp(prefix=f"{stamp}-", dir=root))


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_symlink():
        raise HBServeError(f"output already exists: {path}")
    resolved = path.resolve()
    if resolved.exists():
        raise HBServeError(f"output already exists: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", dir=resolved.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, resolved)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
