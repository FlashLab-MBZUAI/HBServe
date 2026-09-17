# Native SGLang frontend

`python -m hbserve.sglang run` executes timestamped token requests with the
native SGLang scheduler, RadixCache, admission, retraction and KV allocator.
Only forward timing is supplied by the persistent production HBFSim engine.
Every hardware configuration reruns scheduling. Recorded batches are evidence,
not a frozen schedule used for counterfactual timing.

## Runtime and source

The optional runtime is pinned to SGLang
`13d593b6cf885c5c4d50eea88c82b9e28cf5941e`. Base HBServe imports do not import
SGLang or PyTorch. The exercised native runtime uses Python 3.12 on macOS arm64
and Linux x86-64.
Install `requirements/sglang-macos-arm64.txt` into a dedicated environment.
The Linux dependency snapshot is `requirements/sglang-linux.txt`; install a
compatible CPU PyTorch/TorchVision first. Native Qwen3-8B Linux pilots cover
all-HBM, peer weights-first, and tiered KV-first placement.

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
Arrival ties use a unique FIFO index instead of the upstream wall-clock salt,
which can collide while ingesting large traces. `prepare` refreshes older
integration edits only after verifying their recorded file digests.

CPU/dummy-model mode executes the native scheduling runtime without downloading
weights or executing CUDA kernels. Single-worker operation and disabled overlap
scheduling match the pinned upstream simulator's execution model.
Physical forwards can take minutes of host CPU time. The host watchdog uses
`--host-forward-timeout-seconds` (24 hours by default), replacing SGLang's
five-minute serving default. This limit does not enter the simulated clock.

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
- `layout.py` packs weights and KV into capacity-accounted physical extents.
  `--hbm-priority weights-first|kv-first` fills HBM in that order and spills
  to HBF. Priority reserves the **entire configured KV pool**, including the
  native reserved page, rather than moving weights as the live set grows.
  Every layer uses the same HBM slot cutoff. These are fixed homes, with no
  runtime migration. Priority requires explicit `--max-total-tokens` and
  rejects simultaneous `--weight-tier` / `--kv-tier` arguments.
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

Run the commands below from the HBFSim checkout, with both repositories on
`PYTHONPATH`. System profiles are explicitly selected from HBServe; they target
HBFSim `2a59b7f` (see [configuration compatibility](configuration.md)).

Each JSONL row contains integer `arrival_ns`, nonempty `token_ids`, and positive
`output_tokens`. Arrival times must be nondecreasing. Shared token prefixes
produce real radix reuse; a length-only trace cannot supply this information.
An optional unique `request_id` is preserved in native results; otherwise IDs
are assigned in input order. Native token JSONL keeps its first arrival time.

```sh
python -m hbserve.sglang run \
  --sglang-root tmp/sglang-native \
  --simulator build/hbfsim \
  --system ../HBServe/configs/systems/eight-stack-baseline.cfg \
  --system ../HBServe/configs/systems/sglang-small.cfg \
  --model ../HBServe/examples/sglang/tiny-qwen3 \
  --requests ../HBServe/examples/sglang/requests.jsonl \
  --weight-tier hbf --kv-tier hbf --architecture tiered \
  --kv-cache-bytes 8192 --background-writeback-pages 1 \
  --max-total-tokens 128 --page-size 4 \
  --output out/native-sglang
```

### Bailian and Mooncake inputs

The same `run --requests` accepts a canonical bundle directory produced by
HBFSim's `workloads.production_request_trace` importer. To read a raw published
file directly, add `--trace-source-id`: this reuses that importer's pinned
source verification, time-unit conversion and session metadata. Raw slices
are not complete published files; select a slice with `--trace-start` and
`--trace-count` after import instead. `--trace-count` is a maximum number of
consecutive requests. The selected first arrival becomes zero; intervals,
tied-arrival order, input/output lengths and source request IDs are preserved.
Each selection starts with an empty serving cache. Earlier history is not
silently warmed into it. Existing canonical time transformations remain in
effect; this adapter applies no additional time scaling.

From HBFSim with both checkouts on `PYTHONPATH`, this bounded example downloads
the registered Bailian artifact and runs three unchanged source requests:

```sh
python -m workloads.production_request_trace download-source \
  --source-id qwen_bailian_trace_a --output-dir out/public-traces

python -m hbserve.sglang run \
  --sglang-root tmp/sglang-native --simulator build/hbfsim \
  --system ../HBServe/configs/systems/eight-stack-baseline.cfg \
  --system ../HBServe/configs/systems/sglang-small.cfg \
  --model ../HBServe/examples/sglang/tiny-qwen3 \
  --requests out/public-traces/qwen_traceA_blksz_16.jsonl \
  --trace-source-id qwen_bailian_trace_a --trace-start 8946 --trace-count 3 \
  --weight-tier hbm --kv-tier hbf --kv-cache-bytes 131072 \
  --max-total-tokens 16384 --context-length 8192 \
  --page-size 4 --chunked-prefill-size 1024 \
  --output out/bailian-sglang
```

For the Mooncake example, download `mooncake_fast25_conversation`, change
`--requests` to `out/public-traces/FAST25-release/traces/conversation_trace.jsonl`,
use that source ID, and select `--trace-start 6172 --trace-count 3` with a fresh
output directory. These small selections exercise the input path and cache
reuse; they are not representative performance samples. The tiny model is a
functional fixture. Real studies supply the desired dense model configuration
and sufficient memory/context capacity. Requests are never silently shortened
to fit the tiny profile.

Supported registry IDs are `qwen_bailian_trace_a`, `qwen_bailian_trace_b`,
`qwen_bailian_thinking`, `qwen_bailian_coder`, `mooncake_fast25_conversation`,
`mooncake_fast25_toolagent`, and `mooncake_fast25_synthetic`. The last requires
`--allow-synthetic-trace` and retains its synthetic provenance. With a canonical
bundle directory, omit `--trace-source-id`; its manifest supplies the identity.

The traces contain anonymous block hashes, not original token IDs. Conversion
uses a deterministic block-prefix encoding: Bailian blocks have 16 tokens and
Mooncake FAST'25 blocks have 512. Distinct sibling hashes receive different
first tokens; the remaining tokens come from source-namespaced SHAKE-256 bytes.
This preserves the published longest-common-prefix relationships even with a
one-token SGLang page, without accidentally inventing partial-block hits.
An insufficient model vocabulary is an explicit error, not modulo aliasing.
Tokens 0 and 1 are reserved because the pinned dummy decoder emits token 1;
its synthetic output cannot masquerade as a later prompt. The encoding's
codebook belongs to the selected trace. Convert combined selections together
instead of concatenating independently encoded token files.

Original text, unknown overlap inside differing hashes, and reuse of unpublished
generated output cannot be reconstructed. Parent/session fields remain in the
source evidence; requests follow the published arrivals, without adding new
parent-completion dependencies or another chat template.

Each run freezes `inputs/requests.jsonl`, `inputs/trace-records.jsonl`,
`inputs/trace-manifest.json` and `inputs/request-provenance.json`, including source
hashes, selection, arrival origin and encoding. Native request IDs match the
source IDs, and completion checks reconcile arrival and exact input/output
lengths for every request. These files accompany the normal physical receipts.

Placement controls:

| Configuration | Arguments |
| --- | --- |
| HBM resident | `--weight-tier hbm --kv-tier hbm --kv-cache-bytes 0` |
| HBF weights, HBM KV | `--weight-tier hbf --kv-tier hbm --kv-cache-bytes 0` |
| Direct HBF weights/KV | `--weight-tier hbf --kv-tier hbf --architecture peer --kv-cache-bytes 0` |
| Tiered HBF with dirty KV caching | `--architecture tiered --kv-tier hbf --kv-cache-bytes 8192` |
| Weights first, then KV, with HBF overflow | `--hbm-priority weights-first --max-total-tokens N` |
| KV pool first, then weights, with HBF overflow | `--hbm-priority kv-first --max-total-tokens N` |
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
The pinned upstream `request.jsonl` writer rebases its timestamps to the first
arrival. `serving-result.json` restores that known offset for `request_stats`
and records it as `upstream_request_time_origin_ns`, so request timestamps and
physical batches use the input clock. Duration metrics are unchanged.

The compiler uses the existing object-level traffic model and declared roofline
compute (`--compute-tflops`, `--compute-efficiency`). It excludes off-chip scratch
and detailed GPU-cache traffic without an independent kernel trace. Changing
memory timing closes the scheduling loop, but does not establish hardware
latency accuracy. Multiworker collectives, speculative decoding and SGLang
HiCache are outside this dense single-worker adapter.

Outputs include frozen requests/model/system inputs and source/binary hashes,
the physical capacity budget, upstream request/iteration metrics, native batch
and free-slot records, the first actual transaction DAG, per-batch device
receipts, HBF invalidation receipts and final physical wear artifacts. Final
cache writeback and device drain occur after request metrics and are reported
separately. `get_metrics()` never resets physical state. One process runs one
complete benchmark, then closes its HBFSim child.

Batch records retain the request ID, context length, newly written native slots,
input tokens and SHA-256 of the full slot map (little-endian unsigned 32-bit
IDs). They no longer repeat every historical slot on every decode token.
`hbfsim/progress.json` is updated atomically at most every 30 seconds with
completed requests, generated tokens, batches, simulated time and host elapsed
time. Full contexts still feed the native traffic compiler without sampling.

Run the focused physical checks from HBFSim:

```sh
python ../HBServe/tests/test_sglang_backend.py --simulator build/hbfsim -v
python ../HBServe/tests/test_sglang_requests.py -v
```

These checks cover shared prefix liveness, partial-page frees, dead dirty-byte
discard, pressure eviction/reload, detached background writes and source reuse,
nonzero arrival, capacity reservations, dtype geometry and chunk output timing.
They complement actual native scheduler runs; they are not a replacement for
that end-to-end execution.
