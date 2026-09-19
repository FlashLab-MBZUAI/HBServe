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

## Backend compatibility and OCP migration

Current compatibility target: HBFSim **60e3f6669c49a7e8c0a0bd299955de2527ec6a70**.
The profiles were sourced from **2a59b7f13461356d33c00ad1f25fdee4e0bf7fb1**;
their recorded source values are unchanged at the current target. The original
profile source hashes and migration inventory remain intact.
The HBServe base revision **3a6b9a5** shipped older vendor-target profiles that
fail with this backend. This change adopts the corresponding HBFSim OCP v0.7.0
Grade 2 profiles, rather than attempting to reproduce the obsolete model.
`configs/ocp-profile-source.json` records source hashes and **every changed,
added or removed parameter in each of the 11 complete profiles, plus the mini session overlay**. The copied
`configs/parameter-provenance.json` gives upstream evidence and assumption
labels. These are source identities, not hardware validation receipts.

| Parameter | Previous profile | Current profile / meaning |
| --- | --- | --- |
| HBF channels × dies/channel × banks/die | 4 × 4 × 4 | 16 × 1 × 16; 256 banks/stack |
| Blocks/bank (full; mini) | 8192; 800 | 2048; 200, preserving 512; 50 GiB raw/stack |
| `hbf-subarrays-per-plane` | 32 | Removed; backend has one ordered sense resource/bank |
| Media lanes; page-buffer banks/bank | 16; 16 | 1; 2 |
| Read; program latency (ns) | 1000; 95000 | 4000; 75000, upstream assumptions |
| `hbf-program-verify-ns` | 5000 | Removed; no separate configurable verify stage |
| `hbf-hbio-bw` | 1600 GB/s | Removed; interface derived from explicit speed grade 2 |
| ECC decode/encode raw rate | 105.46875 | 101.25 GB/s |
| Channel; TSV rate | 421.875; 1712.5 | 101.25; 1644 GB/s |
| `hbf-page-run-acceleration` | true | Removed obsolete execution switch; no replacement key |
| `hbf-read-buffer-pages` (session mini overlay) | 16 | Removed; backend owns two decoded pages per bank |
| HBM pin rate | 9.6 Gb/s | 8 Gb/s (2048 GB/s/stack), upstream baseline |
| Standards | implicit | Explicit OCP-HBF-0.7.0-2026-08-03 and JEDEC-JESD270-4-2025-04 |

The above are **physical model changes**, not equivalent key renames. Other
values, including stack counts, capacities, thermal assumptions and write-buffer
policy, retain the corresponding profile values. Miniquick changes capacity only
relative to the *new* full profile; it does not retain old-model timing.
`host-hbf-dram-bandwidth-gbps` is also obsolete, but was already absent from the
3a6b9a5 source configs. Controller storage/service follows the selected backend's
HBM contract; this migration does not restore a private controller DRAM model.

| Pair / entry | Startup status | Physical equivalence |
| --- | --- | --- |
| HBServe 3a6b9a5 original profiles + HBFSim 60e3f66 | Rejected (obsolete keys) | No |
| Migrated profiles + updated bundled or target external client + HBFSim 60e3f66 | Bounded config/session tests pass | Not equivalent to original profiles |
| Migrated profiles + bundled client from 6f5a94e + HBFSim 60e3f66 | New transaction census, tier accounting and wear-v2 receipts rejected | Previous protocol fix needs this update |
| Migrated profiles + historical HBFSim versions | Not certified; pin matching source/configs | Not assumed |
| Historical results + original binary/config/trace | Retained evidence, not rerun here | Unchanged files; original scope only |
| Zero-HBM `0h8f` execution | Still unsupported by controller-HBM contract | Not repaired by this change |

Flat serving profiles and `systems/` profiles now select identical values for
matching shapes. SGLang examples explicitly select HBServe's `systems/` base
and `sglang-small.cfg` overlay; commands in `docs/sglang.md` run from HBFSim.
The native coarse path remains the default. No simulator or serving algorithm
is changed. The bundled Python simulation-session client is updated to the
selected backend contract: scalar read receipts with conservation checks,
`pending_block_transitions`, `raw-physical` mapping, resolved HBM burst and
controller buffer reservations (including scratch/GC), zone-managed image
validation, and logical page invalidation / wear-output hooks needed by SGLang. The old bundled
client rejects the new read receipt even after every config is fixed. Running
from the HBServe directory selects this bundled client ahead of PYTHONPATH;
therefore configuration-only changes do not fix standalone HBServe. No legacy
field fallback or old physical model is reintroduced.

The current backend also emits a `HOST_DRAM` transaction census and an explicit
`host_dram` accounting slot, including zero/null entries when CPU DRAM is disabled.
The bundled client validates these fields on completion, invalidation, checkpoint,
crash and close, and accepts an explicit `host_dram_config` attachment. Host DRAM
and external backing retain separate transaction, page-run and transport accounting.
This client interface does not add a new serving placement policy.

Wear snapshots use schema v2: per-block erase counts must agree with physical
state totals and verified accounting, and the pending-work counters must agree
with the quiescence flag. Observing a snapshot allows pending writes and does
not implicitly drain them. The previous client/schema is not accepted as an
equivalent fallback. The supported backend is the explicit commit above.

Run the focused checks with explicit paths (no editable install required):

```sh
python tests/test_config_compatibility.py \
  --hbfsim-root /path/to/HBFSim --simulator /path/to/hbfsim
python -B tests/test_session_compatibility.py --simulator /path/to/hbfsim
```

Without `--simulator`, the session suite runs portable receipt acceptance and
corruption checks in CI; native HBM-only, Host-DRAM/NVMe coexistence and pending
wear/checkpoint tests require the explicit executable. Local compatibility
validation runs with both the bundled and target HBFSim clients.

These checks verify config acceptance, profile provenance values, capacity
invariants and window config composition. Synthetic session checks only certify
software wiring. They do not validate inference performance, GPU cache accuracy,
phase-specific hardware errors, or a new model/context. Freeze new configurations
for any rerun; simulated time, traffic and placement may change, and old fits,
calibration coefficients and reported errors must not automatically carry over.
Do not rewrite historical experiment JSON, configs, manifests or results.
