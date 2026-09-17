"""Run timestamped token requests through native SGLang and HBFSim."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys

from hbfsim_client import SimulationSessionError

from .backend import HardwareSession
from .model import DTYPE_BYTES, DTYPE_NAMES
from .prepare import prepare, sha, verify
from .requests import bind_request_ids, load_requests, request_stats_on_input_clock, validate_request_stats


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="prepare a separate pinned upstream checkout")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--reference", type=Path, help="optional local SGLang repository for a separate worktree")
    run = commands.add_parser("run", help="execute complete native requests")
    run.add_argument("--sglang-root", type=Path, required=True)
    run.add_argument("--simulator", type=Path, required=True)
    run.add_argument("--system", type=Path, action="append", required=True)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--requests", type=Path, required=True,
                     help="token JSONL, canonical trace bundle, or pinned raw trace with --trace-source-id")
    run.add_argument("--trace-source-id", help="HBFSim registry source ID for a raw Bailian/Mooncake file")
    run.add_argument("--trace-start", type=int, default=0, help="first canonical request index (default: 0)")
    run.add_argument("--trace-count", type=int, help="maximum consecutive trace requests; selected arrivals rebase to zero")
    run.add_argument("--allow-synthetic-trace", action="store_true", help="allow the labeled Mooncake synthetic source")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--weight-tier", choices=["hbm", "hbf", "external"])
    run.add_argument("--kv-tier", choices=["hbm", "hbf", "external"])
    run.add_argument("--hbm-priority", choices=["weights-first", "kv-first"],
                     help="pack the specified class into HBM first; spill to HBF; reserves the full KV pool")
    run.add_argument("--architecture", choices=["peer", "tiered"], default="tiered")
    run.add_argument("--kv-cache-bytes", type=int, default=0)
    run.add_argument("--background-writeback-pages", type=int, default=2)
    run.add_argument("--transfer-chunk-bytes", type=int, default=65536)
    run.add_argument("--workspace-bytes", type=int, default=1048576)
    run.add_argument("--max-total-tokens", type=int)
    run.add_argument("--max-running-requests", type=int, default=4)
    run.add_argument("--chunked-prefill-size", type=int, default=8)
    run.add_argument("--max-prefill-tokens", type=int, default=16384)
    run.add_argument("--page-size", type=int, default=1)
    run.add_argument("--context-length", type=int)
    run.add_argument("--disable-radix-cache", action="store_true")
    run.add_argument("--enable-mixed-chunk", action="store_true")
    run.add_argument("--schedule-policy", choices=["fcfs", "lpm", "random", "dfs-weight", "lof"], default="fcfs")
    run.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    run.add_argument("--kv-cache-dtype", choices=["auto", *DTYPE_BYTES], default="auto")
    run.add_argument("--scheduler-overhead-ns", type=float, default=0)
    run.add_argument("--host-forward-timeout-seconds", type=float, default=86400,
                     help="host watchdog per physical forward; does not affect simulated time (default: 24 hours)")
    run.add_argument("--compute-tflops", type=float, default=100)
    run.add_argument("--compute-efficiency", type=float, default=0.5)
    return p


def run(args):
    source = args.sglang_root.resolve()
    upstream = verify(source)
    hf = json.loads((args.model / "config.json").read_text())
    dtype = args.dtype if args.dtype != "auto" else hf.get("dtype", hf.get("torch_dtype", "bfloat16"))
    kv_dtype = dtype if args.kv_cache_dtype == "auto" else args.kv_cache_dtype
    if kv_dtype != dtype and kv_dtype not in {"bfloat16", "fp8_e4m3"}:
        raise ValueError("the pinned CPU runtime accepts KV in model dtype, bfloat16 or fp8_e4m3")
    if not math.isfinite(args.scheduler_overhead_ns) or args.scheduler_overhead_ns < 0:
        raise ValueError("scheduler-overhead-ns must be finite and nonnegative")
    if not math.isfinite(args.host_forward_timeout_seconds) or args.host_forward_timeout_seconds <= 0:
        raise ValueError("host-forward-timeout-seconds must be finite and positive")
    request_input = load_requests(args.requests, hf["vocab_size"], source_id=args.trace_source_id,
        start=args.trace_start, count=args.trace_count, allow_synthetic=args.allow_synthetic_trace)
    records = request_input.records
    for name in ("max_running_requests", "chunked_prefill_size", "max_prefill_tokens", "page_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    config = {"model": hf, "dtype": dtype, "kv_dtype": kv_dtype,
        "simulator": str(args.simulator.resolve()), "system_configs": [str(p.resolve()) for p in args.system],
        "compute": {"peak_tflops": args.compute_tflops, "efficiency": args.compute_efficiency},
        **{name: getattr(args, name) for name in ("weight_tier", "kv_tier", "hbm_priority", "architecture", "kv_cache_bytes",
            "background_writeback_pages", "transfer_chunk_bytes", "workspace_bytes", "page_size", "max_total_tokens")}}
    layout = HardwareSession(config, initialize=False)
    config["max_total_tokens"] = layout.max_total_tokens
    context = args.context_length or min(hf.get("max_position_embeddings", layout.max_total_tokens), layout.max_total_tokens)
    for record in records:
        required = len(record["token_ids"]) + record["output_tokens"]
        if required > context:
            raise ValueError(f"request {record['request_id']} needs {required} prompt+output tokens, "
                             f"exceeding the context/capacity limit {context}; use sufficient model/context "
                             "and memory capacity or select a suitable trace window")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / "inputs"
    inputs.mkdir()
    model_path = inputs / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(hf, indent=2))
    frozen_system = []
    for i, path in enumerate(args.system):
        dest = inputs / f"system-{i}.cfg"
        shutil.copyfile(path, dest)
        frozen_system.append(str(dest))
    config["system_configs"] = frozen_system
    request_input.freeze(inputs)
    hardware_path = out / "hardware.json"
    hardware_path.write_text(json.dumps(config, indent=2))
    (out / "capacity-budget.json").write_text(json.dumps(layout.budget, indent=2))
    frontend_root = Path(__file__).resolve().parents[1]
    source_paths = [*Path(__file__).parent.glob("*.py"), frontend_root / "compiler.py", frontend_root / "contracts.py"]
    source_files = {str(p.relative_to(frontend_root)): sha(p) for p in source_paths}
    manifest = {"upstream": upstream, "frontend_files": source_files,
        "requests_sha256": sha(inputs / "requests.jsonl"), "request_source": request_input.provenance,
        "model_sha256": sha(args.model / "config.json"),
        "simulator_sha256": sha(args.simulator), "arguments": {k: str(v) if isinstance(v, Path) else
            [str(p) for p in v] if k == "system" else v for k, v in vars(args).items()},
        "scope": "Native scheduler and allocator; request origin and token encoding in request_source; modeled GPU and memory timing"}
    (out / "inputs.json").write_text(json.dumps(manifest, indent=2))
    # Ensure spawned workers load the same maintained physical client as this
    # process, even when an older HBServe checkout has its own bundled client.
    import hbfsim_client
    import hbserve
    paths = list(dict.fromkeys([str(Path(hbfsim_client.__file__).resolve().parents[1]),
        str(Path(hbserve.__file__).resolve().parents[1]), str(source), str(source / "python"),
        str(source / "python/sglang/kernels/aot/python"), str(source / "tools/sglang-simulator/src")]))
    sys.path[:0] = paths
    os.environ["PYTHONPATH"] = os.pathsep.join(paths + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
    os.environ["SGLANG_USE_CPU_ENGINE"] = "1"
    os.environ["SGLANG_SIMULATOR_OUTPUT_DIR"] = str(out)
    os.environ["SGLANG_SIMULATOR_OUTPUT_MODE"] = "OFFLINE"
    simulator_config = {"platform": {"accelerator": {"name": "hbfsim", "vendor": "simulation",
        "hbm_capacity_gb": layout.budget["capacity_bytes"]["hbm"] / 1e9, "hbm_bandwidth_gb": 0},
        "num_device_per_node": 1},
        "predictor": {"name": "hbfsim", "hardware_config": str(hardware_path),
                      "cpu_overhead_ns": args.scheduler_overhead_ns},
        "scheduler": {"tp_size": 1, "ep_size": 1, "dp_size": 1,
            "data_type": DTYPE_NAMES[dtype], "kv_cache_data_type": DTYPE_NAMES[kv_dtype]}}
    simulator_config_path = out / "simulator.json"
    simulator_config_path.write_text(json.dumps(simulator_config, indent=2))
    os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = str(simulator_config_path)
    from sglang_simulator.dataset import GenericRequest, SimpleDataset
    from sglang_simulator.simulation.benchmark import BenchmarkConfig
    from sglang_simulator.simulation.sglang.launch_server import apply_simulator_defaults
    apply_simulator_defaults(argparse.Namespace(), [])
    from benchmark.simulator.bench_runner import SGLangBenchmarkRunner
    from sglang.srt.server_args import ServerArgs
    dataset = SimpleDataset(reqs=[GenericRequest(token_ids=r["token_ids"], input_length=len(r["token_ids"]),
        output_length=r["output_tokens"], custom_params={"created_time": r["arrival_ns"] / 1e9}) for r in records])
    native_kv_dtype = "bf16" if kv_dtype == "bfloat16" else "auto" if kv_dtype == dtype else kv_dtype
    runner = SGLangBenchmarkRunner(server_args=ServerArgs(model_path=str(model_path),
        load_format="dummy", device="cpu", random_seed=0, dtype=dtype, kv_cache_dtype=native_kv_dtype,
        max_total_tokens=layout.max_total_tokens, max_running_requests=args.max_running_requests,
        chunked_prefill_size=args.chunked_prefill_size, max_prefill_tokens=args.max_prefill_tokens,
        page_size=args.page_size, context_length=context, schedule_policy=args.schedule_policy,
        enable_mixed_chunk=args.enable_mixed_chunk,
        skip_tokenizer_init=True, disable_radix_cache=args.disable_radix_cache,
        watchdog_timeout=args.host_forward_timeout_seconds,
        enable_hierarchical_cache=False, sampling_backend="pytorch"))
    bind_request_ids(runner.engine, records)
    try:
        metrics = runner.benchmark(BenchmarkConfig(ignore_request_timestamp=False), dataset=dataset)
        result = {"metrics": metrics, "request_stats": request_stats_on_input_clock(
                    runner.get_request_stats(), records[0]["arrival_ns"]),
                  "upstream_request_time_origin_ns": records[0]["arrival_ns"],
                  "iteration_stats": runner.get_iteration_stats(), "manifest": manifest}
        (out / "serving-result.json").write_text(json.dumps(result, indent=2))
        if metrics is None or metrics.get("completed") != len(records):
            raise RuntimeError("SGLang did not complete every request; inspect the worker log")
        validate_request_stats(records, result["request_stats"])
        physical = json.loads((out / "hbfsim/result.json").read_text())
        if not physical["finalized"] or physical["dirty_cache_bytes"]:
            raise RuntimeError("HBFSim did not finalize and drain its cache")
        print(json.dumps({"completed": metrics["completed"], "batches": physical["batches"],
                          "result": str(out / "serving-result.json"), "physical": str(out / "hbfsim/result.json")}, indent=2))
    finally:
        runner.shutdown()


def main():
    p = parser()
    args = p.parse_args()
    try:
        if args.command == "prepare":
            print(json.dumps(prepare(args.source, args.reference), indent=2))
        else:
            run(args)
    except (ValueError, FileNotFoundError, SimulationSessionError) as error:
        p.error(str(error))


if __name__ == "__main__":
    main()
