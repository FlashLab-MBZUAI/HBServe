"""HBServe: one workload frontend for closed-loop serving and fixed windows.

Request path: catalog descriptor -> request source -> token-budgeted mixed
iterations with paged KV -> placement-independent object DAG -> object-class
placement (weights by tier, hot KV in HBM, cold KV in HBF or external) ->
persistent HBFSim session, with roofline compute by default.

The package deliberately separates request/model semantics from the mapped
memory protocol.  ``HBServeCompiler`` creates placement-independent
object accesses; placement backends turn those objects into HBM, HBF, or
external-memory transactions. The optional ``hbserve.sglang`` frontend uses
native SGLang scheduling and slot allocation with its own physical mapper.

``hbserve.windows`` supplies matched, deterministic memory-only windows for
controlled topology comparisons without request scheduling or compute timing.
"""

from hbserve.compiler import HBServeCompiler
from hbserve.contracts import (
    BatchSlice,
    CanonicalServingBatch,
    LayerSpec,
    LinearTimingProvider,
    MemoryObject,
    MemoryOnlyTimingProvider,
    ModelSpec,
    RequestSpec,
    RequestTrace,
    RooflineTimingProvider,
    RouterDecision,
    RouterTrace,
    ScheduledBatch,
    SchedulerPolicy,
    HBServeError,
    TraceProvenance,
)
from hbserve.engine import BatchExecution, HBServeEngine
from hbserve.placement import KvPlacement, PlacementSpec, HBServePlacement
from hbserve.synthetic import (
    HotsetZipfRouter,
    SyntheticRequestConfig,
    generate_requests,
)

__version__ = "0.1.0a2"

__all__ = [
    "BatchSlice",
    "BatchExecution",
    "CanonicalServingBatch",
    "KvPlacement",
    "LayerSpec",
    "LinearTimingProvider",
    "MemoryObject",
    "MemoryOnlyTimingProvider",
    "ModelSpec",
    "HotsetZipfRouter",
    "PlacementSpec",
    "RequestSpec",
    "RequestTrace",
    "RooflineTimingProvider",
    "RouterDecision",
    "RouterTrace",
    "ScheduledBatch",
    "SchedulerPolicy",
    "HBServeEngine",
    "HBServePlacement",
    "HBServeCompiler",
    "HBServeError",
    "TraceProvenance",
    "SyntheticRequestConfig",
    "generate_requests",
    "__version__",
]
