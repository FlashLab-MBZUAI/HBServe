#!/usr/bin/env python3
"""One workload frontend with explicit closed-loop and fixed-window inputs.

``python -m hbserve run`` takes a model (catalog descriptor or
``hbserve.model`` JSON), a system config with optional overlays, a request
source (synthetic spec or request trace), and a placement preset or JSON,
derives the placement capacities from the system config, runs the closed
loop, writes every input and the receipted result into a fresh timestamped
directory, and prints a short headline. Alternatively, ``--experiment``
selects a matched fixed memory window, without request scheduling or compute.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]

from hbfsim_client.simulation_session import (  # noqa: E402
    ResolvedSystemConfig,
    SimulationSessionError,
)
from hbfsim_client.provenance import add_allow_dirty_argument
from hbserve.catalog import load_model_any  # noqa: E402
from hbserve.compiler import HBServeCompiler  # noqa: E402
from hbserve.contracts import (  # noqa: E402
    HBServeError,
    ModelSpec,
    RequestTrace,
    SchedulerPolicy,
    TimingProvider,
    canonical_sha256,
)
from hbserve.engine import HBServeEngine  # noqa: E402
from hbserve.hbfsim import HbfSimExecutor  # noqa: E402
from hbserve.io import (  # noqa: E402
    REQUEST_TRACE_SCHEMA,
    RUN_CONFIG_SCHEMA,
    SYNTHETIC_REQUEST_SCHEMA,
    create_run_directory,
    load_json_object,
    load_router,
    placement_from_dict,
    run_config_from_dict,
    synthetic_request_config_from_dict,
    write_json_atomic,
)
from hbserve.placement import (  # noqa: E402
    KvPlacement,
    HBServePlacement,
    PlacementSpec,
)
from hbserve.synthetic import generate_requests  # noqa: E402


EXPERIMENT_SCHEMA = {
    "name": "hbserve.experiment",
    "version": 2,
}
DEFAULT_PEAK_TFLOPS = 989.0  # H100 SXM dense BF16 tensor peak
DEFAULT_EFFICIENCY = 0.6
DEFAULT_SCHEDULER = {
    "max_batch_requests": 256,
    "max_batch_tokens": 8192,
    "prefill_chunk_tokens": 2048,
}
# Placement presets: weight tier for every model and the cold KV tier.
PLACEMENT_PRESETS: dict[str, dict[str, Any]] = {
    "weights-hbf-kv-hbm": {"weights": "hbf", "kv_cold": None},
    "all-hbm": {"weights": "hbm", "kv_cold": None},
    "weights-hbf-kv-hbf-cold": {"weights": "hbf", "kv_cold": "hbf"},
}
HBM_RUNTIME_RESERVE_FRACTION = 0.05


def _artifact(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise HBServeError(
            f"input artifact must be a regular non-symlink file: {path}"
        )
    resolved = path.resolve()
    if not resolved.is_file():
        raise HBServeError(
            f"input artifact must be a regular non-symlink file: {resolved}"
        )
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _align_down(value: int, alignment: int) -> int:
    return value // alignment * alignment


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def derive_placement(
    *,
    system: ResolvedSystemConfig,
    models: Mapping[str, ModelSpec],
    preset: str,
) -> PlacementSpec:
    """Derive placement capacities from the system config for one preset."""

    try:
        rule = PLACEMENT_PRESETS[preset]
    except KeyError as error:
        raise HBServeError(
            f"unknown placement preset {preset!r}; choose one of "
            f"{', '.join(sorted(PLACEMENT_PRESETS))} or pass a placement JSON"
        ) from error
    geometry = system.hbf_geometry
    hbm_capacity = system.hbm_capacity_bytes
    if system.logical_hbf_capacity_bytes is None:
        raise HBServeError("placement requires geometry resolved by the simulator")
    hbf_capacity = system.logical_hbf_capacity_bytes
    external_capacity = 0
    external_page = 4096
    if system.values.get("external-backing-kind") is not None:
        identity = system.external_backing_identity
        external_capacity = int(identity["capacity_bytes"])
        external_page = int(identity["page_size_bytes"])
    weights_tier = str(rule["weights"])
    if weights_tier == "hbf" and hbf_capacity == 0:
        raise HBServeError("preset places weights in HBF but the system has none")
    alignment = 4096
    return PlacementSpec(
        hbm_capacity_bytes=hbm_capacity,
        hbf_capacity_bytes=(
            hbf_capacity
            if weights_tier == "hbf" or rule["kv_cold"] == "hbf"
            else 0
        ),
        external_capacity_bytes=external_capacity,
        hbm_runtime_reserve_bytes=_align_up(
            int(hbm_capacity * HBM_RUNTIME_RESERVE_FRACTION), alignment
        ),
        hbm_model_cache_bytes=0,
        model_weight_tiers={model_id: weights_tier for model_id in models},
        object_tier_overrides={},
        hbm_alignment_bytes=alignment,
        hbf_page_size_bytes=geometry.page_size_bytes,
        external_page_size_bytes=external_page,
        kv_block_tokens=16,
        kv_placement=KvPlacement(hot="hbm", cold=rule["kv_cold"]),
    )


def hbm_stripe_bytes(system: ResolvedSystemConfig) -> int:
    """Bytes one pass over every HBM pseudo-channel covers (the map stripe).

    Mirrors the engine: pseudo-channels x interleave bytes, where an unset
    ``hbm-interleave-bytes`` selects the largest multiple of the burst that
    does not exceed 256 B.  The value only steers host-time-neutral
    alignment and splitting in the placement; a wrong guess costs speed,
    never correctness.
    """

    pseudo_channels = (
        system.integer("hbm-stacks")
        * system.integer("hbm-channels")
        * system.integer("hbm-pseudo-channels")
    )
    burst = system.hbm_burst_bytes
    try:
        interleave = system.integer("hbm-interleave-bytes", minimum=0)
    except SimulationSessionError:
        interleave = 0
    if interleave == 0:
        interleave = max(burst, 256 // burst * burst)
    return pseudo_channels * interleave


def load_requests_any(path: Path) -> RequestTrace:
    """Load a request trace or generate one from a synthetic spec."""

    document = load_json_object(path, "requests")
    schema = document.get("schema")
    if schema == SYNTHETIC_REQUEST_SCHEMA:
        return generate_requests(synthetic_request_config_from_dict(document))
    if schema == REQUEST_TRACE_SCHEMA:
        from hbserve.io import load_request_trace

        return load_request_trace(path)
    raise HBServeError(
        f"{path} is neither a {SYNTHETIC_REQUEST_SCHEMA['name']} spec nor a "
        f"{REQUEST_TRACE_SCHEMA['name']} trace"
    )


def default_run_config(
    *,
    timing: str,
    peak_tflops: float,
    efficiency: float,
    prefetch_depth: int,
) -> dict[str, Any]:
    if timing == "roofline":
        timing_section: dict[str, Any] = {
            "type": "roofline",
            "prefetch_depth": prefetch_depth,
        }
        compute: dict[str, Any] | None = {
            "peak_tflops": peak_tflops,
            "efficiency": efficiency,
        }
    elif timing == "memory_only":
        timing_section = {"type": "memory_only", "prefetch_depth": prefetch_depth}
        compute = None
    else:
        raise HBServeError("default timing must be roofline or memory_only")
    return {
        "schema": dict(RUN_CONFIG_SCHEMA),
        "scheduler": dict(DEFAULT_SCHEDULER),
        "timing": timing_section,
        "compute": compute,
    }


def _format_bytes(value: float) -> str:
    for unit, scale in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if value >= scale:
            return f"{value / scale:.2f} {unit}"
    return f"{int(value)} B"


def _format_ns(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value >= 1e9:
        return f"{value / 1e9:.3f} s"
    if value >= 1e6:
        return f"{value / 1e6:.3f} ms"
    if value >= 1e3:
        return f"{value / 1e3:.3f} us"
    return f"{value:.0f} ns"


def headline_lines(
    experiment: Mapping[str, Any],
    *,
    result_path: Path,
    placement_label: str,
) -> list[str]:
    """At most ten lines that say what ran and what came out."""

    hbserve = experiment["hbserve"]
    summary = hbserve["summary"]
    timing = hbserve["timing"]
    scheduler = hbserve["scheduler"]
    models = hbserve["models"]
    trace = experiment["inputs"]["request_trace"]
    session = experiment["final_execution"]["session"]
    devices = session["final_measurement"]["device_workload_totals"]
    percentiles = summary["latency_percentiles"]
    lines = [
        (
            f"hbserve run: {summary['requests']} requests, "
            f"{trace['prompt_tokens']} prompt + {summary['output_tokens']} output "
            f"tokens; model {', '.join(sorted(models))}; "
            f"system {experiment['inputs']['system_label']}"
        )
    ]
    provider = timing["provider"]
    if timing["model"] == "roofline":
        compute_label = (
            f"peak {provider['peak_tflops']:g} TFLOPS x {provider['efficiency']:.2f} "
            f"efficiency, prefetch depth {timing['prefetch_depth']}"
        )
    elif timing["model"] == "gpu_calibrated":
        compute_label = "calibrated GPU operators; inferred compute/memory overlap"
    elif timing["model"] == "linear":
        compute_label = (
            f"{provider['fixed_ns_per_layer']:g} ns + "
            f"{provider['ns_per_token_per_layer']:g} ns/token per layer"
        )
    else:
        compute_label = "no compute; memory critical path only"
    lines.append(
        f"timing model: {timing['model']} ({compute_label}); placement: "
        f"{placement_label}"
    )
    kinds = scheduler["iterations_by_kind"]
    migration = scheduler["kv_migration_bytes"]
    lines.append(
        f"iterations: {scheduler['batches']} (prefill {kinds['prefill']}, decode "
        f"{kinds['decode']}, mixed {kinds['mixed']}); preemptions: "
        f"{summary['preemptions']}; KV migrated out/in: "
        f"{_format_bytes(migration['swap_out'])} / "
        f"{_format_bytes(migration['swap_in'])}"
    )
    if summary["includes_compute"]:
        ttft = percentiles["ttft_ns"] or {}
        tpot = percentiles["tpot_ns"] or {}
        lines.append(
            f"TTFT mean {_format_ns(summary['mean_ttft_ns'])} "
            f"(p50 {_format_ns(ttft.get('p50'))}, p95 {_format_ns(ttft.get('p95'))}); "
            f"TPOT mean {_format_ns(summary['mean_tpot_ns'])} "
            f"(p50 {_format_ns(tpot.get('p50'))}, p95 {_format_ns(tpot.get('p95'))})"
        )
        throughput = summary["output_token_throughput_per_second"]
        lines.append(
            "output throughput "
            + ("n/a" if throughput is None else f"{throughput:,.1f} tokens/s")
            + f"; span {_format_ns(summary['duration_ns'])}"
        )
    else:
        lines.append(
            "memory critical path: first token mean "
            f"{_format_ns(summary['mean_memory_critical_path_first_token_ns'])}, "
            "per token mean "
            f"{_format_ns(summary['mean_memory_critical_path_per_token_ns'])} "
            "(not TTFT/TPOT: no compute in this timing model)"
        )
        lines.append(f"span {_format_ns(summary['duration_ns'])}")
    hbm = devices.get("hbm") or {}
    hbf = devices.get("hbf") or {}
    external = devices.get("external") or {}
    lines.append(
        f"bytes: HBM R {_format_bytes(hbm.get('read_bytes', 0))} / W "
        f"{_format_bytes(hbm.get('write_bytes', 0))}; HBF R "
        f"{_format_bytes(hbf.get('logical_read_bytes', 0))} / W "
        f"{_format_bytes(hbf.get('logical_write_bytes', 0))}; external R "
        f"{_format_bytes(external.get('read_bytes', 0))} / W "
        f"{_format_bytes(external.get('write_bytes', 0))}"
    )
    logical_writes = int(hbf.get("logical_write_bytes", 0))
    physical_writes = int(hbf.get("physical_write_bytes", 0))
    waf = (
        "n/a (no HBF writes)"
        if logical_writes == 0
        else f"{physical_writes / logical_writes:.3f}"
    )
    warnings: list[str] = []
    if not summary["includes_compute"]:
        warnings.append("memory-only timing reports no TTFT/TPOT")
    if summary["preemptions"]:
        warnings.append(f"{summary['preemptions']} preemption(s)")
    if not hbserve["eligibility"]["source_qualified_request_trace"]:
        warnings.append("synthetic requests")
    lines.append(
        f"HBF WAF: {waf}; warnings: {'; '.join(warnings) if warnings else 'none'}"
    )
    lines.append(f"output: {result_path}")
    return lines


def run_experiment(
    *,
    simulator: Path,
    system_config_paths: Sequence[Path],
    models: Mapping[str, ModelSpec],
    request_trace: RequestTrace,
    router: Any,
    placement_spec: PlacementSpec,
    policy: SchedulerPolicy,
    timing: TimingProvider,
    prefetch_depth: int,
    input_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the closed loop and return the receipted experiment document."""

    try:
        system = ResolvedSystemConfig.load(list(system_config_paths))
    except SimulationSessionError as error:
        raise HBServeError(str(error)) from error
    compiler = HBServeCompiler(
        models=models,
        request_trace=request_trace,
        router=router,
        timing=timing,
        prefetch_depth=prefetch_depth,
    )
    placement = HBServePlacement(
        models=models,
        spec=placement_spec,
        hbf_geometry=system.hbf_geometry,
        hbm_stripe_bytes=hbm_stripe_bytes(system),
    )
    executor = HbfSimExecutor(
        simulator_path=simulator,
        system_config_paths=list(system_config_paths),
        placement=placement,
    )
    try:
        hbserve_result = HBServeEngine(
            models=models,
            request_trace=request_trace,
            compiler=compiler,
            executor=executor,
            policy=policy,
        ).run()
    finally:
        executor.close()
    final_execution = executor.final_receipt()
    source = final_execution["session"]
    if (
        source["simulator_executable"] != input_artifacts["simulator"]
        or source["system_configs"] != input_artifacts["system_configs"]
    ):
        raise HBServeError(
            "HBFSim execution identity differs from the pre-run artifacts"
        )
    experiment = {
        "schema": EXPERIMENT_SCHEMA,
        "execution_mode": "closed_loop",
        "result": "pass",
        "inputs": dict(input_artifacts),
        "hbserve": hbserve_result,
        "final_execution": final_execution,
    }
    experiment["experiment_sha256"] = canonical_sha256(experiment)
    return experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hbserve run",
        description=(
            "Run closed-loop requests (--requests) or a fixed memory window "
            "(--experiment) through HBFSim. Scale is selected by the input "
            "configs, not by a different generator."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--requests",
        type=Path,
        help="closed loop: hbserve.synthetic_requests or hbserve.request_trace JSON",
    )
    source.add_argument(
        "--experiment",
        type=Path,
        help="fixed window: explicit experiment JSON (miniquick or full scale); "
        "no compute, request admission feedback, or TTFT/TPOT",
    )
    serving = parser.add_argument_group("closed-loop requests only")
    serving.add_argument(
        "--model",
        type=Path,
        action="append",
        help="hbserve.public_model descriptor or "
        "hbserve.model JSON; required with --requests; repeat for multi-model serving",
    )
    serving.add_argument(
        "--system", type=Path, help="system config; required with --requests"
    )
    serving.add_argument(
        "--overlay",
        type=Path,
        action="append",
        help="overlay config applied after --system (repeatable)",
    )
    serving.add_argument("--router", type=Path, help="MoE router trace or synthetic router")
    serving.add_argument(
        "--placement",
        help="preset (" + ", ".join(sorted(PLACEMENT_PRESETS)) + ") or placement JSON; "
        "required with --requests",
    )
    serving.add_argument(
        "--run-config",
        type=Path,
        help="hbserve.run_config JSON; default: token-budgeted mixed iterations "
        "with roofline timing",
    )
    serving.add_argument(
        "--timing",
        choices=("roofline", "memory_only"),
        help="default: roofline; cannot combine timing knobs with --run-config",
    )
    serving.add_argument(
        "--peak-tflops", type=float, help=f"default: {DEFAULT_PEAK_TFLOPS}"
    )
    serving.add_argument(
        "--efficiency", type=float, help=f"default: {DEFAULT_EFFICIENCY}"
    )
    serving.add_argument("--prefetch-depth", type=int, help="default: 1")
    serving.add_argument("--prefix-cache-bytes", type=int, help="bounded full-block prefix cache in the shared HBM KV pool")
    serving.add_argument("--prefix-cache-ttl-ns", type=float, help="optional prefix entry lifetime from publication; requires a nonzero cache budget")
    window = parser.add_argument_group("fixed-window experiments only")
    window.add_argument(
        "--topologies",
        help="comma-separated topology IDs to execute; default: all declared rows",
    )
    window.add_argument(
        "--preflight-only",
        action="store_true",
        help="generate and validate the entire window/matrix without a simulator",
    )
    add_allow_dirty_argument(window)
    parser.add_argument("--simulator", type=Path, help="path to a compatible external HBFSim executable")
    parser.add_argument("--out", type=Path, required=True, help="output root directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    timing_defaults = {
        "timing": "roofline",
        "peak_tflops": DEFAULT_PEAK_TFLOPS,
        "efficiency": DEFAULT_EFFICIENCY,
        "prefetch_depth": 1,
    }
    serving_options = (
        "model",
        "system",
        "overlay",
        "router",
        "placement",
        "run_config",
        "prefix_cache_bytes",
        "prefix_cache_ttl_ns",
        *timing_defaults,
    )
    if args.experiment is not None:
        invalid = [name for name in serving_options if getattr(args, name) is not None]
        if invalid:
            parser.error(
                "--experiment cannot use closed-loop options: "
                + ", ".join("--" + name.replace("_", "-") for name in invalid)
            )
        if args.preflight_only and args.topologies is not None:
            parser.error(
                "--preflight-only validates all rows; --topologies is execution-only"
            )
        from hbserve.windows.run import run_window

        if not args.preflight_only and args.simulator is None:
            parser.error("--simulator is required for physical execution")
        return run_window(args, parser)
    if args.topologies is not None or args.preflight_only or args.allow_dirty:
        parser.error(
            "--topologies, --preflight-only, and --allow-dirty require --experiment"
        )
    missing = [
        name for name in ("model", "system", "placement")
        if getattr(args, name) is None
    ]
    if missing:
        parser.error("--requests requires " + ", ".join("--" + name for name in missing))
    if args.run_config is not None and any(
        getattr(args, name) is not None for name in timing_defaults
    ):
        parser.error("--run-config cannot be combined with timing/prefetch overrides")
    for name, value in timing_defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.simulator is None:
        parser.error("--simulator is required for physical execution")
    try:
        system_paths = [args.system, *(args.overlay or [])]
        models = {}
        model_artifacts = []
        for path in args.model:
            model = load_model_any(path)
            if model.model_id in models:
                raise HBServeError(f"duplicate model id {model.model_id}")
            models[model.model_id] = model
            model_artifacts.append(_artifact(path))
        request_trace = load_requests_any(args.requests)
        router = None if args.router is None else load_router(args.router)
        system = ResolvedSystemConfig.load(system_paths).resolve(args.simulator)
        placement_path: Path | None = None
        if args.placement in PLACEMENT_PRESETS:
            placement_spec = derive_placement(
                system=system, models=models, preset=args.placement
            )
            placement_label = args.placement
        else:
            placement_path = Path(args.placement)
            placement_spec = placement_from_dict(
                load_json_object(placement_path, "serving placement")
            )
            placement_label = str(placement_path)
        if args.prefix_cache_bytes is not None or args.prefix_cache_ttl_ns is not None:
            placement_spec = replace(
                placement_spec,
                prefix_cache_bytes=(placement_spec.prefix_cache_bytes if args.prefix_cache_bytes is None else args.prefix_cache_bytes),
                prefix_cache_ttl_ns=(placement_spec.prefix_cache_ttl_ns if args.prefix_cache_ttl_ns is None else args.prefix_cache_ttl_ns),
            )
        if args.run_config is not None:
            run_config = load_json_object(args.run_config, "serving run config")
        else:
            run_config = default_run_config(
                timing=args.timing,
                peak_tflops=args.peak_tflops,
                efficiency=args.efficiency,
                prefetch_depth=args.prefetch_depth,
            )
        policy, timing, prefetch_depth = run_config_from_dict(run_config)
        if timing.timing_model == "gpu_calibrated":
            models = {key: timing.profile.bind_model(model, timing.model_bindings[key])
                      for key, model in models.items()}
            if args.placement in PLACEMENT_PRESETS:
                placement_spec = replace(derive_placement(system=system,models=models,preset=args.placement),
                    prefix_cache_bytes=placement_spec.prefix_cache_bytes,
                    prefix_cache_ttl_ns=placement_spec.prefix_cache_ttl_ns,
                    kv_block_tokens=timing.profile.document['kv_block_tokens'])
            elif placement_spec.kv_block_tokens != timing.profile.document['kv_block_tokens']:
                raise HBServeError("placement page size differs from the calibrated attention backend")
        input_artifacts = {
            "simulator": _artifact(args.simulator),
            "system_configs": [_artifact(path) for path in system_paths],
            "system_label": " + ".join(
                path.resolve().relative_to(ROOT).as_posix()
                if path.resolve().is_relative_to(ROOT)
                else str(path)
                for path in system_paths
            ),
            "models": model_artifacts,
            "requests": _artifact(args.requests),
            "request_trace": {
                "sha256": request_trace.digest,
                "requests": len(request_trace.requests),
                "prompt_tokens": sum(
                    request.prompt_tokens for request in request_trace.requests
                ),
                "output_tokens": sum(
                    request.output_tokens for request in request_trace.requests
                ),
            },
            "router": None if args.router is None else _artifact(args.router),
            "placement": (
                {"preset": args.placement, "sha256": placement_spec.digest}
                if placement_path is None
                else _artifact(placement_path)
            ),
            "run_config": (
                _artifact(args.run_config)
                if args.run_config is not None
                else {"default": True, "sha256": canonical_sha256(run_config)}
            ),
        }
        output_directory = create_run_directory(args.out)
        for model in models.values():
            write_json_atomic(
                output_directory / f"model-{model.model_id}.json", model.canonical()
            )
        write_json_atomic(output_directory / "requests.json", request_trace.canonical())
        write_json_atomic(output_directory / "placement.json", placement_spec.canonical())
        write_json_atomic(output_directory / "run-config.json", run_config)
        experiment = run_experiment(
            simulator=args.simulator,
            system_config_paths=system_paths,
            models=models,
            request_trace=request_trace,
            router=router,
            placement_spec=placement_spec,
            policy=policy,
            timing=timing,
            prefetch_depth=prefetch_depth,
            input_artifacts=input_artifacts,
        )
        result_path = output_directory / "result.json"
        write_json_atomic(result_path, experiment)
        lines = headline_lines(
            experiment, result_path=result_path, placement_label=placement_label
        )
        (output_directory / "headline.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    except (OSError, HBServeError, SimulationSessionError) as error:
        print(f"hbserve run failed: {error}", file=sys.stderr)
        return 2
    print("\n".join(lines))
    return 0
