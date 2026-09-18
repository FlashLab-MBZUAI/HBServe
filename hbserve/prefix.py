"""Content-addressed, immutable full-block prefix caching in the shared KV pool."""

from collections import OrderedDict
from dataclasses import dataclass

from hbserve.contracts import canonical_sha256


def block_keys(request, model_digest, block_tokens):
    if request.token_ids is None:
        return ()
    parent = canonical_sha256({"model": model_digest, "salt": request.cache_salt,
                               "block_tokens": block_tokens, "position": 0})
    keys = []
    for begin in range(0, request.prompt_tokens - block_tokens + 1, block_tokens):
        parent = canonical_sha256({"parent": parent, "tokens": request.token_ids[begin:begin + block_tokens]})
        keys.append(parent)
    return tuple(keys)


@dataclass(frozen=True)
class PrefixEntry:
    blocks: tuple[int, ...]
    expires_ns: float | None
    pending: bool = False


class PrefixCache:
    def __init__(self, pool, capacity_bytes, ttl_ns):
        self.pool = pool
        self.capacity_bytes = capacity_bytes
        self.ttl_ns = ttl_ns
        self.entries = OrderedDict()
        self.bytes = 0
        self.stats = {"lookups": 0, "hit_requests": 0, "hit_tokens": 0,
                      "inserted_blocks": 0, "evicted_blocks": 0, "expired_blocks": 0}

    def evict(self, key, *, expired=False):
        entry = self.entries.pop(key)
        self.pool.release(entry.blocks)
        self.bytes -= len(entry.blocks) * self.pool.block_bytes
        self.stats["expired_blocks" if expired else "evicted_blocks"] += 1

    def expire(self, now_ns):
        if self.ttl_ns is not None:
            for key, entry in tuple(self.entries.items()):
                if entry.expires_ns is not None and entry.expires_ns <= now_ns:
                    self.evict(key, expired=True)

    def lookup(self, keys, max_blocks, block_tokens, now_ns):
        self.expire(now_ns)
        self.stats["lookups"] += 1
        found = []
        for key in keys[:max_blocks]:
            entry = self.entries.get(key)
            if entry is None:
                break
            self.entries.move_to_end(key)
            self.pool.retain(entry.blocks)
            found.append(entry.blocks)
        self.stats["hit_requests"] += int(bool(found))
        self.stats["hit_tokens"] += len(found) * block_tokens
        return found

    def publish(self, keys, blocks, complete_blocks, now_ns, *, pending=False):
        self.expire(now_ns)
        for ordinal, key in enumerate(keys[:complete_blocks]):
            if key in self.entries:
                entry = self.entries[key]
                if entry.pending and not pending:
                    self.entries[key] = PrefixEntry(entry.blocks,
                        None if self.ttl_ns is None else now_ns+self.ttl_ns)
                self.entries.move_to_end(key)
                continue
            selected = tuple(layer[ordinal] for layer in blocks)
            required = len(selected) * self.pool.block_bytes
            if required > self.capacity_bytes:
                continue
            while self.bytes + required > self.capacity_bytes:
                self.evict(next(iter(self.entries)))
            self.pool.retain(selected)
            self.entries[key] = PrefixEntry(selected,
                None if pending or self.ttl_ns is None else now_ns + self.ttl_ns,pending)
            self.bytes += required
            self.stats["inserted_blocks"] += 1

    def release_order(self, keys):
        # vLLM returns a completed request's blocks tail-first to the free
        # queue. Evicting suffixes first preserves usable partial prefixes;
        # evicting block zero would make every retained suffix unreachable.
        for key in reversed(keys):
            entry = self.entries.get(key)
            if entry is not None and all(self.pool.references.get(b) == 1 for b in entry.blocks):
                self.entries.move_to_end(key)

    def receipt(self):
        return {"enabled": self.capacity_bytes > 0, "tier": "hbm", "policy": "full_block_content_hash_lru",
                "capacity_bytes": self.capacity_bytes, "resident_bytes": self.bytes,
                "resident_blocks": len(self.entries), "ttl_ns": self.ttl_ns, **self.stats,
                "storage": "reference_counted_immutable_blocks_shared_with_active_KV",
                "durable": False, "partial_block_reuse": False,
                "metadata_cpu_time": "not_modeled"}
