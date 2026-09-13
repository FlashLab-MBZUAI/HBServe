"""Sparse, packed reference counts for allocated KV blocks."""

from array import array
from collections.abc import Iterator, MutableMapping


class BlockReferences(MutableMapping[int, int]):
    """Zero denotes an unallocated block; empty chunks release their storage."""

    _CHUNK = 1024

    def __init__(self) -> None:
        self._chunks: dict[int, array] = {}
        self._length = 0

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, block: int) -> int:
        chunk = self._chunks.get(block // self._CHUNK)
        value = 0 if chunk is None else chunk[block % self._CHUNK]
        if not value:
            raise KeyError(block)
        return value

    def __setitem__(self, block: int, value: int) -> None:
        if not 0 < value < 2**64:
            raise ValueError("KV reference count must be a positive uint64")
        index, offset = divmod(block, self._CHUNK)
        chunk = self._chunks.get(index)
        if chunk is None:
            chunk = array("Q", [0]) * (self._CHUNK + 1)
            self._chunks[index] = chunk
        if not chunk[offset]:
            chunk[-1] += 1
            self._length += 1
        chunk[offset] = value

    def __delitem__(self, block: int) -> None:
        index, offset = divmod(block, self._CHUNK)
        chunk = self._chunks.get(index)
        if chunk is None or not chunk[offset]:
            raise KeyError(block)
        chunk[offset] = 0
        chunk[-1] -= 1
        self._length -= 1
        if not chunk[-1]:
            del self._chunks[index]
        if not self._length:
            self._chunks.clear()

    def __iter__(self) -> Iterator[int]:
        for index, chunk in self._chunks.items():
            for offset in range(self._CHUNK):
                if chunk[offset]:
                    yield index * self._CHUNK + offset
