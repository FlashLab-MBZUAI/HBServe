# Native SGLang frontend

`python -m hbserve.sglang run` executes timestamped token requests with the
native SGLang scheduler, RadixCache, admission, retraction and KV allocator.
Only forward timing is supplied by the persistent production HBFSim engine.
Every hardware configuration reruns scheduling. Recorded batches are evidence,
not a frozen schedule used for counterfactual timing.

## Runtime and source

The optional runtime is pinned to SGLang
`13d593b6cf885c5c4d50eea88c82b9e28cf5941e`. Base HBServe imports do not import
SGLang or PyTorch. The exercised native runtime uses Python 3.12 on macOS arm64.
Install `requirements/sglang-macos-arm64.txt` into a dedicated environment.
The Linux dependency snapshot is `requirements/sglang-linux.txt`; install a
compatible CPU PyTorch/TorchVision first. Linux execution of this new adapter
has not been verified by the local macOS run.

Use the maintained `hbfsim_client` from the HBFSim source checkout, ahead of
HBServe on `PYTHONPATH`. This matters when the HBServe checkout contains an
older bundled client. Build the matching HBFSim executable from that checkout.

```sh
# From HBFSim, with HBServe in the neighboring directory:
export PYTHONPATH="$PWD:../HBServe${PYTHONPATH:+:$PYTHONPATH}"
python -m hbserve.sglang prepare --source tmp/sglang-native
```

`prepare` fetches a separate source checkout and patches only predictor
registration, native batch snapshots, simulator timing/profile hooks, and
macOS CPU initialization. `--reference /path/to/sglang` creates a separate
worktree at the exact revision without copying that checkout's edits. It can
reuse HBFUltra's source repository; HBFUltra's Python modules are not runtime
dependencies. Preparation refuses to overwrite pre-existing upstream edits.

CPU/dummy-model mode executes the native scheduling runtime without downloading
weights or executing CUDA kernels. Single-worker operation and disabled overlap
scheduling match the pinned upstream simulator's execution model.

## Ownership and physical execution

- `adapter.py` captures pre-forward tokens and the real per-request slot map,
  including mixed batches. It observes actual Token/Paged allocator frees,
  including unused page padding. Request completion alone does not invalidate KV.
- `model.py` derives dense Llama, Qwen2, Qwen3 and full-context Mistral geometry.
  It reuses `HBServeCompiler` with actual input tokens instead of surrogate IDs.
  BF16/FP16/FP32 weights, KV in model precision, and explicit BF16/FP8-E4M3 KV
  storage are supported by the pinned CPU runtime. Weight
  quantization, MoE and compressed/windowed attention are rejected explicitly.
- `backend.py` maps immutable model objects and native slots to HBM, logical
  HBF, or configured external backing. It does not run HBServe's scheduler or
  block allocator. The fixed-window and synthetic HBServe frontends remain
  useful for their deterministic controlled experiments.
- Tiered access uses bounded HBM staging and actual per-stack D2D traffic for
  HBF. Peer access uses direct physical transactions. An optional page cache
  has finite HBM capacity, LRU replacement, partial valid/dirty ranges and
  write-no-read allocation. Eviction transfers dirty bytes through HBM, links
  and backing before reusing the cache slot.
- `--background-writeback-pages` bounds dirty pages submitted each batch.
  Foreground completion is the core's blocking frontier; cache reuse and reads
  carry dependencies on outstanding transfers. The same media/link resources
  serve foreground and background work. The core frontier also respects the
  latest issued dependent transaction; it is not an arbitrary early timestamp.
- Freeing a slot removes only its sectors. Shared backing pages stay live until
  every resident slot is freed. Whole-page HBF invalidation uses the core's
  explicit issued-IO fence, whose delay enters the next callback. External
  frees update liveness but do not invent SSD TRIM support.

Capacity includes weights, the HBM cache, two transfer buffers, explicit
workspace, reserved controller HBM, layer alignment and SGLang's reserved page.
The resulting token capacity controls SGLang admission. `--max-total-tokens`
can impose a smaller page-aligned cap. Physical slot numbering is preserved;
the byte layout is a declared layer-major packed K/V layout, not a measured
GPU kernel layout.

## Requests and execution

Each JSONL row contains integer `arrival_ns`, nonempty `token_ids`, and positive
`output_tokens`. Arrival times must be nondecreasing. Shared token prefixes
produce real radix reuse; a length-only trace cannot supply this information.

```sh
python -m hbserve.sglang run \
  --sglang-root tmp/sglang-native \
  --simulator build/hbfsim \
  --system configs/systems/eight-stack-baseline.cfg \
  --system configs/systems/sglang-small.cfg \
  --model ../HBServe/examples/sglang/tiny-qwen3 \
  --requests ../HBServe/examples/sglang/requests.jsonl \
  --weight-tier hbf --kv-tier hbf --architecture tiered \
  --kv-cache-bytes 8192 --background-writeback-pages 1 \
  --max-total-tokens 128 --page-size 4 \
  --output out/native-sglang
```

Placement controls:

| Configuration | Arguments |
| --- | --- |
| HBM resident | `--weight-tier hbm --kv-tier hbm --kv-cache-bytes 0` |
| HBF weights, HBM KV | `--weight-tier hbf --kv-tier hbm --kv-cache-bytes 0` |
| Direct HBF weights/KV | `--weight-tier hbf --kv-tier hbf --architecture peer --kv-cache-bytes 0` |
| Tiered HBF with dirty KV caching | `--architecture tiered --kv-tier hbf --kv-cache-bytes 8192` |
| External KV | `--kv-tier external`, plus an HBFSim external-backing system overlay |

Other options include chunked prefill, `--enable-mixed-chunk` for mixed
prefill/decode batches, admission limits, page size, scheduler
policy, radix disable, model/KV dtype and explicit target CPU overhead. Run
`python -m hbserve.sglang run --help` for their names. A fresh output directory
is required for each execution.

## Clocks, output and evidence

The predictor returns `(foreground_finish_ns - scheduler_start_ns) / 1e9`.
Idle gaps are advanced inside the physical session before lifecycle processing,
so they are not charged twice. Host predictor execution time is recorded as
`host_cpu_overhead`; only the explicit `--scheduler-overhead-ns` advances the
target clock. Its default zero is an uncalibrated hypothesis. A nonzero first
arrival remains nonzero at the first physical callback.

The compiler uses the existing object-level traffic model and declared roofline
compute (`--compute-tflops`, `--compute-efficiency`). It excludes off-chip scratch
and detailed GPU-cache traffic without an independent kernel trace. Changing
memory timing closes the scheduling loop, but does not establish hardware
latency accuracy. Multiworker collectives, speculative decoding and SGLang
HiCache are outside this dense single-worker adapter.

Outputs include frozen requests/model/system inputs and source/binary hashes,
the physical capacity budget, upstream request/iteration metrics, native batch
and free-slot snapshots, the first actual transaction DAG, per-batch device
receipts, HBF invalidation receipts and final physical wear artifacts. Final
cache writeback and device drain occur after request metrics and are reported
separately. `get_metrics()` never resets physical state. One process runs one
complete benchmark, then closes its HBFSim child.

Run the focused physical checks from HBFSim:

```sh
python ../HBServe/tests/test_sglang_backend.py --simulator build/hbfsim -v
```

These checks cover shared prefix liveness, partial-page frees, dead dirty-byte
discard, pressure eviction/reload, detached background writes and source reuse,
nonzero arrival, capacity reservations, dtype geometry and chunk output timing.
They complement actual native scheduler runs; they are not a replacement for
that end-to-end execution.
