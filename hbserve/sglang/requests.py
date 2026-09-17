"""Native tokens and source-bound Bailian/Mooncake requests for SGLang.

Anonymous blocks become surrogate tokens, not reconstructed model content.
Distinct branches diverge at the first token of a source block, so even a
one-token RadixCache page cannot manufacture sub-block prefix hits.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import tempfile

from .model import positive_int
from .prepare import sha


@dataclass
class RequestInput:
    records: list[dict]
    provenance: dict
    source_records: tuple[dict, ...] = ()
    source_manifest: dict | None = None

    def freeze(self, directory: Path) -> None:
        with (directory / "requests.jsonl").open("w") as stream:
            for row in self.records:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        (directory / "request-provenance.json").write_text(json.dumps(self.provenance, indent=2) + "\n")
        if self.source_manifest is not None:
            from workloads.production_request_trace import canonical_json_bytes
            (directory / "trace-manifest.json").write_bytes(canonical_json_bytes(self.source_manifest) + b"\n")
            with (directory / "trace-records.jsonl").open("wb") as stream:
                for row in self.source_records:
                    stream.write(canonical_json_bytes(row) + b"\n")


def validate_requests(records: list[dict], vocab_size: int) -> None:
    if not records:
        raise ValueError("request trace must not be empty")
    previous, identities = -1, set()
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f"request {index} must be a JSON object")
        arrival, tokens, output = row.get("arrival_ns"), row.get("token_ids"), row.get("output_tokens")
        if (type(arrival) is not int or arrival < 0 or arrival < previous or not isinstance(tokens, list)
                or not tokens or any(type(t) is not int or not 0 <= t < vocab_size for t in tokens)
                or type(output) is not int or output < 1):
            raise ValueError(f"invalid request {index}; need ordered arrival_ns, token_ids and output_tokens")
        rid = row.setdefault("request_id", f"native:{index}")
        if not isinstance(rid, str) or not rid or rid in identities:
            raise ValueError(f"request {index} needs a unique, nonempty request_id")
        identities.add(rid)
        previous = arrival


def encode_trace(trace, vocab_size: int, *, start=0, count=None, allow_synthetic=False) -> RequestInput:
    """Encode a validated canonical trace; lengths and source hashes stay exact.

    A lexical traversal assigns a different leading token to every sibling
    edge in the block-prefix trie. Only the preceding path is retained, not
    a second in-memory copy of the full trie. Block bodies are deterministic
    SHAKE-256 bytes. IDs 0 and 1 are reserved: the pinned simulator emits 1 on
    decode, which must never masquerade as unpublished prompt content.
    """
    positive_int(vocab_size, "vocab_size", 3)
    positive_int(start, "trace-start", 0)
    if count is not None:
        positive_int(count, "trace-count")
    if not trace.manifest["source"]["production_eligible"] and not allow_synthetic:
        raise ValueError("synthetic trace requires --allow-synthetic-trace")
    stop = len(trace.records) if count is None else min(len(trace.records), start + count)
    if start >= stop:
        raise ValueError("trace selection is empty; trace-start must be below the request count")
    selected = trace.records[start:stop]
    origin = selected[0]["arrival_ns"]
    block = trace.prefix_block_tokens
    encoded = [None] * len(selected)
    previous, codes, maximum_branching = [], [], 0
    for index in sorted(range(len(selected)), key=lambda i: selected[i]["prefix_hash_ids"]):
        row = selected[index]
        hashes = row["prefix_hash_ids"]
        common = 0
        while common < min(len(previous), len(hashes)) and previous[common] == hashes[common]:
            common += 1
        if common < len(hashes):
            rank = codes[common] + 1 if common < len(previous) else 2
            codes = codes[:common] + [rank] + [2] * (len(hashes) - common - 1)
        else:
            codes = codes[:common]
        maximum_branching = max(maximum_branching, max(codes) - 1)
        if max(codes) >= vocab_size:
            raise ValueError(f"trace needs at least {max(codes)+1} vocabulary entries to preserve exact "
                             "block-prefix branching; use a matching larger-vocabulary model or a smaller trace slice")
        tokens = []
        for position, (block_hash, leading_token) in enumerate(zip(hashes, codes)):
            valid = min(block, row["input_tokens"] - position * block)
            seed = f"hbserve.sglang.block.v1:{trace.source_namespace}:{block_hash}".encode()
            body = hashlib.shake_256(seed).digest(4 * (valid - 1))
            tokens.append(leading_token)
            tokens.extend(2 + int.from_bytes(body[i:i+4], "little") % (vocab_size - 2)
                          for i in range(0, len(body), 4))
        encoded[index] = {"request_id": row["request_id"], "source_index": row["source_index"],
            "arrival_ns": row["arrival_ns"] - origin, "token_ids": tokens,
            "output_tokens": row["output_tokens"]}
        previous = hashes
    validate_requests(encoded, vocab_size)
    return RequestInput(encoded, {
        "format": "canonical_production_trace", "source_namespace": trace.source_namespace,
        "source_manifest_sha256": trace.manifest_sha256, "source_requests_sha256": trace.requests_sha256,
        "source": trace.manifest["source"], "selection": {"start": start, "stop": stop,
            "arrival_origin_ns": origin, "time_scale": "1/1", "cold_start": True},
        "encoding": {"name": "lexical_block_prefix_v1", "vocab_size": vocab_size,
            "prefix_block_tokens": block, "reserved_token_ids": [0, 1],
            "maximum_branching": maximum_branching,
            "scope": "Preserves published ordered block-prefix identity and input lengths; "
                     "unknown sub-block overlap and generated-output reuse are not reconstructed. "
                     "The codebook is local to this selected trace; convert combined slices together."},
        "request_count": len(encoded), "input_tokens": sum(len(r["token_ids"]) for r in encoded),
        "output_tokens": sum(r["output_tokens"] for r in encoded),
    }, selected, trace.manifest)


def load_requests(path: Path, vocab_size: int, *, source_id=None, start=0, count=None,
                  allow_synthetic=False) -> RequestInput:
    path = Path(path)
    if source_id is None and not path.is_dir():
        if start or count is not None or allow_synthetic:
            raise ValueError("trace selection options require a canonical bundle or --trace-source-id")
        with path.open() as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        if records and isinstance(records[0], dict) and "hash_ids" in records[0]:
            raise ValueError("raw Bailian/Mooncake input requires --trace-source-id")
        validate_requests(records, vocab_size)
        return RequestInput(records, {"format": "native_token_jsonl", "source_sha256": sha(path),
                                      "request_count": len(records)})
    try:
        from workloads.production_request_trace import load_trace
        from workloads.production_request_trace.registry import load_source_spec
        from workloads.production_request_trace.qwen_bailian import import_qwen_bailian
        from workloads.production_request_trace.mooncake_fast25 import import_mooncake_fast25
    except ImportError as error:
        raise ValueError("trace input requires the HBFSim checkout on PYTHONPATH") from error
    if path.is_dir():
        if source_id is not None:
            raise ValueError("a canonical bundle already identifies its source; omit --trace-source-id")
        trace = load_trace(path)
    else:
        spec = load_source_spec(source_id)
        # Reuse the SHA-pinned importers; never accept an unverified raw slice
        # as though it were the complete published artifact.
        with tempfile.TemporaryDirectory(prefix="hbserve-trace-") as temporary:
            destination = Path(temporary) / "canonical"
            if spec.importer == "qwen_bailian_v1":
                trace = import_qwen_bailian(path, destination, source_id=source_id)
            elif spec.importer == "mooncake_fast25_v1":
                trace = import_mooncake_fast25(path, destination, source_id=source_id,
                                               allow_synthetic=allow_synthetic)
            else:
                raise ValueError(f"unsupported trace importer {spec.importer}")
    return encode_trace(trace, vocab_size, start=start, count=count, allow_synthetic=allow_synthetic)


def bind_request_ids(engine, records):
    """The pinned benchmark dispatches in dataset order; retain source IDs."""
    generate = engine.async_generate
    pending = iter(records)

    def with_identity(*args, **kwargs):
        row = next(pending)
        return generate(*args, **kwargs, rid=row["request_id"])

    engine.async_generate = with_identity


def request_stats_on_input_clock(stats, first_arrival_ns):
    """Undo the pinned profile writer's first-arrival rebase, not execution time."""
    origin = first_arrival_ns / 1e9
    fields = ("created_time", "queue_start", "queue_end", "last_event_time")
    return [{**row, **{key: row[key] + origin for key in fields}} for row in stats]


def validate_request_stats(records, stats):
    expected = {row["request_id"]: row for row in records}
    if len(stats) != len(expected) or {s["rid"] for s in stats} != set(expected):
        raise RuntimeError("native results do not match the input request identities")
    for stat in stats:
        row = expected[stat["rid"]]
        arrival = row["arrival_ns"] / 1e9
        if (not math.isclose(stat["created_time"], arrival, rel_tol=0, abs_tol=1e-12)
                or stat["input_length"] != len(row["token_ids"])
                or stat["output_length"] != row["output_tokens"]
                or len(stat["gen_token_latencies"]) != row["output_tokens"]):
            raise RuntimeError(f"native request {stat['rid']} changed its arrival or input/output length")
        if (stat["queue_start"] < arrival-1e-12 or stat["queue_end"] < stat["queue_start"]-1e-12
                or stat["last_event_time"] < arrival-1e-12
                or any(not math.isfinite(t) or t < -1e-12 for t in stat["gen_token_latencies"])):
            raise RuntimeError("native request timestamps violate causality")
