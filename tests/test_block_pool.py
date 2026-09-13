import gc
import hashlib
import json
import random
import tracemalloc
import unittest

from hbserve.contracts import HBServeError, canonical_sha256
from hbserve.placement import _BlockPool


class BlockPoolTests(unittest.TestCase):
    def test_lowest_free_shared_references_against_independent_model(self):
        rng = random.Random(20260913)
        pool = _BlockPool("hbm", 4096, 4096 + 317 * 64, 64)
        references = {}
        for _ in range(10000):
            free = sorted(set(range(317)) - references.keys())
            action = rng.randrange(3)
            if action == 0 and free:
                count = rng.randrange(min(len(free), 20) + 1)
                blocks = pool.allocate(count)
                self.assertEqual(blocks, free[:count])
                references.update((block, 1) for block in blocks)
            elif references:
                blocks = rng.sample(list(references), min(len(references), rng.randrange(1, 12)))
                if action == 1:
                    pool.retain(blocks)
                    for block in blocks:
                        references[block] += 1
                else:
                    pool.release(blocks)
                    for block in blocks:
                        references[block] -= 1
                        if references[block] == 0:
                            del references[block]
            self.assertEqual(dict(pool.references), references)
            self.assertEqual(pool.free_blocks, 317 - len(references))

    def test_capacity_and_completed_history_do_not_allocate_per_block(self):
        gc.collect()
        tracemalloc.start()
        try:
            pool = _BlockPool("hbm", 0, 100000000 * 64, 64)
            self.assertLess(tracemalloc.get_traced_memory()[0], 65536)
            for _ in range(4):
                blocks = pool.allocate(100000)
                # Keep the returned block table in the measurement: active
                # references must fit alongside the actual caller's addresses.
                self.assertLess(tracemalloc.get_traced_memory()[0], 6 * 1024 * 1024)
                pool.release(blocks[::2])
                pool.release(blocks[1::2])
                del blocks
                self.assertLess(tracemalloc.get_traced_memory()[0], 256 * 1024)
            self.assertEqual(pool.allocate(4), [0, 1, 2, 3])
        finally:
            tracemalloc.stop()

    def test_invalid_reference_operations(self):
        pool = _BlockPool("hbf", 0, 10 * 64, 64)
        with self.assertRaises(HBServeError):
            pool.allocate(11)
        blocks = pool.allocate(3)
        pool.retain([blocks[0]])
        pool.release(blocks)
        self.assertEqual(pool.references[0], 1)
        with self.assertRaises(HBServeError):
            pool.release([1])
        with self.assertRaises(HBServeError):
            pool.retain([1])
        pool.release([0])
        self.assertEqual(pool.allocate(4), [0, 1, 2, 3])

    def test_incremental_digest_preserves_canonical_bytes(self):
        value = {"unicode": "你好", "nested": [{"float": -0.0, "flag": True, "x": None}] * 10000}
        expected = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode("utf-8")).hexdigest()
        self.assertEqual(canonical_sha256(value), expected)


if __name__ == "__main__":
    unittest.main()
