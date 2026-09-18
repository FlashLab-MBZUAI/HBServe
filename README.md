# HBServe

HBServe is an evidence-aware workload compiler for studying the memory side of
LLM serving. It turns model descriptors and request streams into continuous-
batching iterations, paged-KV lifecycle events, and byte-exact memory
transactions. A compatible HBFSim executable can then close the loop with
physical memory completion times.

The optional [native SGLang frontend](docs/sglang.md) preserves the production
scheduler, RadixCache and token allocator and uses HBFSim as its forward-time
backend. It supports HBM/HBF/external placement, a finite HBM KV cache,
background writeback and allocator-driven KV invalidation. Run it with
`python -m hbserve.sglang`; the optional runtime has separate dependencies.
Its request input accepts token JSONL, canonical production-trace bundles, or
pinned Bailian/Mooncake files with `--trace-source-id`.

The same `hbserve run` command also accepts fixed memory-window experiments.
Miniquick and full-scale use the same generator, model ledger, and remappers;
their model, population, and system profiles differ, not their implementation.

HBServe is a workload simulator frontend, not an inference engine. Its primary
goal is to make every realism claim inspectable: synthetic inputs stay labeled
synthetic, source artifacts are hashed, model bytes are derived by explicit
formulas, and each result carries a machine-readable capability boundary.

## What is modeled

- Dense, MoE, and multi-model request streams.
- Token-budgeted continuous batching with chunked prefill and decode.
- Incremental 16-token paged KV allocation, migration, preemption, and
  recomputation.
- Content-addressed full-block prefix reuse with a bounded, reference-counted
  HBM cache, LRU pressure eviction, and optional TTL.
- Object-exact weight, embedding, LM-head, block-table, and KV traffic.
- HBM, HBF, and external-memory placement with byte-conservation receipts.
- Calibrated paged GPU operators, roofline, memory-only, or linear timing.
  `gpu_calibrated` preserves ragged requests, actual MoE expert counts and
  compute/memory overlap; its A100 validation scope travels with the profile.
- A persistent HBFSim session whose completion frontier schedules the next
  iteration.
- Matched fixed-window topology comparisons: prefill growth, decode-only, and
  source-anchored mixed windows, with no compute timing or request feedback.

HBServe currently models one device and no collective/network timing. It does
not claim kernel-level cache behavior or hardware-calibrated end-to-end latency
unless those are supplied and validated by a future trace/calibration backend.
Run `hbserve capabilities` to inspect the exact current boundary.

## Install

HBServe requires Python 3.10 or newer and has no runtime Python dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
hbserve --version
hbserve capabilities
```

## Quick start

Model conversion and request generation work without HBFSim:

```bash
mkdir -p out
hbserve model models/llama31-8b-w8-kv-bf16.json \
  --output out/llama31-8b.json

hbserve generate \
  --config examples/quickstart-requests.json \
  --output out/requests.json
```

Physical execution requires a separately built compatible `hbfsim` binary:

```bash
hbserve run \
  --simulator /path/to/HBFSim/build/hbfsim \
  --model models/llama31-8b-w8-kv-bf16.json \
  --system configs/4hbm-4hbf-miniquick.cfg \
  --requests examples/quickstart-requests.json \
  --placement weights-hbf-kv-hbm \
  --out out/quickstart
```

Every closed-loop run creates a new unique timestamped directory. It records canonical inputs,
SHA-256 identities, mapped-byte conservation, per-iteration schedules,
per-request latency rows, physical completions, eligibility flags, and a short
headline. Existing results are never overwritten.

Generate a miniquick window and validate its topology matrix without a simulator:

```bash
hbserve run --experiment configs/windows/miniquick-serving.json \
  --preflight-only --out out/window-check
```

For physical execution, replace `--preflight-only` with
`--simulator /path/to/hbfsim --topologies all-hbm,4h4f`. Select
`configs/windows/full-scale-serving.json` for the full-scale MoE population.
Full-scale execution can be expensive; check the generated workload first.
Use `--allow-dirty` for exploratory runs from a modified or non-Git installation.
See [Fixed windows](docs/windows.md) for modes, profiles, and interpretation.
See [Prefix caching](docs/prefix-caching.md) for identity, lifecycle, synthetic
reuse inputs, and the volatile-cache boundary.

## Recovery scope

The `hbserve.recovery` API is used by HBFSim's Q5 bounded resident-KV controls
in `studies/hbf_persistence/hbserve_recovery.py`. It models committed-boundary
HBF media and KV recovery, not scheduler resume or persistent prefix caching.
HBServe tests its integrity and payload-accounting helpers without a simulator;
HBFSim's separate recovery tests execute small native sessions.

## Why the workload is credible

HBServe separates four questions that benchmarks often conflate:

1. **Input realism:** request arrivals, prompt/output lengths, token IDs, and
   MoE routes are either source-qualified traces or explicitly synthetic.
2. **Semantic realism:** in closed-loop mode, the scheduler and paged-KV state machine operate on
   individual requests and tokens rather than a fixed bandwidth loop.
3. **Traffic realism:** model dimensions and precision produce an auditable
   per-object byte ledger; mapping must conserve those bytes exactly.
4. **Timing realism:** timing is named by backend. Roofline and simulation are
   models, not measurements; hardware claims require calibration and holdout
   validation.

See [Realism and trace integration](docs/realism.md) for the concrete path from
a serving profiler to cache-line address/arrival traces, and
[Design](docs/design.md) for the current contracts.

## Repository layout

- `hbserve/`: request semantics, scheduler, compiler, placement, and CLI.
- `hbserve/windows/`: deterministic fixed-population windows and address remapping.
- `hbfsim_client/`: the minimal persistent-session protocol used by the HBFSim
  backend.
- `models/`: public, source-attributed architecture descriptors.
- `examples/`: deterministic synthetic fixtures and quick-start inputs.
- `configs/`: exploratory simulator profiles; see
  [Configuration](docs/configuration.md).
- `tests/`: contract, scheduler, accounting, and optional physical integration
  tests.

## Test

```bash
python3 -B tests/test_hbserve.py
python3 -B tests/test_windows.py
python3 -B tests/test_recovery.py
python3 -B tests/test_hbserve.py --simulator /path/to/HBFSim/build/hbfsim
python3 -B tests/test_windows.py --simulator /path/to/HBFSim/build/hbfsim
```

Without `--simulator`, the serving/window suites are self-contained and skip physical tests.
With it, they exercise closed-loop serving and a tiny fixed-window execution;
the tests do not execute the full miniquick or full-scale physical matrix.

## Maturity and license

HBServe is alpha software. Schemas are versioned, but compatibility is not yet
promised. The project is available under the [MIT License](LICENSE).
