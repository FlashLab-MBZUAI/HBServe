# Configuration provenance

The checked-in system configurations are exploratory simulator profiles, not
vendor performance guarantees. They are included so users can exercise
placement and protocol behavior with a compatible HBFSim build.

`4hbm-4hbf.cfg` and its `-miniquick` variant combine a four-stack HBM domain
with four HBF stacks. Link rates, queue depths, controller timing, flash-media
timing, overprovisioning, and thermal parameters are modeling assumptions. The
mini profile reduces capacity and host work for fast integration tests; it is
not a smaller hardware product claim.

`eight-stack-baseline.cfg`, `simulation-session-mini.cfg`, `cxl-memory.cfg`,
and `nvme-ssd.cfg` exercise the baseline HBM domain, persistent session
protocol, and external backing alternatives.

`windows/` contains matched miniquick/full-scale experiments. Its paths resolve
relative to each experiment JSON. `systems/` and `overlays/` provide their full
topology matrix, capacity controls, and thermal assumptions. The original flat
profiles remain small request-serving examples; they are not alternate workload
generators. See [Fixed windows](windows.md) for scale and completion boundaries.

The `overlays/backing/calibrated/dana-a100-*` coefficients are inherited from
HBFSim's A100 host-offload fits, not portable DRAM/NVMe specifications. The
upstream raw measurements and validation receipts are not shipped here. This
standalone package therefore treats them as example parameterizations, not
self-contained calibration evidence; validate or replace them for your setup.

For publishable results, freeze the complete config set, hash it in the run
receipt, cite a source or calibration artifact for every physical parameter,
and validate against a holdout workload. Parameters without such evidence must
remain labeled assumptions or sensitivity variables.

## Calibrated GPU operators

A run config may select `timing.type = "gpu_calibrated"`, with an absolute
`profile` JSON path, `model_bindings` (model ID to `8b`, `70b`, or `235b`) and
`prefetch_depth: 0`; set `compute` to null. The CLI binds the packed weight
layout before placement and uses the profile's 256-token KV blocks. Supply
sufficient HBM runtime scratch. The provider reports the profile hash and
measured validation errors in the result; `calibrated` does not imply a
production SLO guarantee.

The corresponding profile/evidence lives in the sibling HBFSim repository at
`evidence/hardware/gpu_operators/`. The v3 runtime rejects old contiguous-KV v2
profiles. Its native client requires dynamic memory-span barrier support.

Scheduler semantics match the tested vLLM 0.26.0 synchronous FCFS configuration:
running requests (including unfinished prefills) in admission order, then
waiting requests; incremental KV allocation; complete prefix blocks shareable
within a batch; youngest-running-request preemption. Prefix release frees
suffixes first, and active entries remain protected during capacity eviction.
This does not claim vLLM async scheduling, speculative decoding, full-prompt
reservation, TP/EP or every scheduler version.
