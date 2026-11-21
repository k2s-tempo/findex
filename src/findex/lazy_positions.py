"""
Позиції токенів, що розгортаються з буфера файлу індексу лише на вимогу (лаба 8).

Раніше load_array одразу перетворював усі позиції на списки Python: для корпусу x10 це 10,7 млн
об'єктів int, ~380 МБ і майже половина часу завантаження. Позиції потрібні лише фразам у лапках, тож
тепер вони лишаються байтами буфера, а словник {doc_id: [позиції]} для терміна будується при першому
зверненні й кешується (кеш обмежений: позиції частого терміна це сотні тисяч чисел).

Поводиться як звичайний словник term -> {doc_id: [позиції]}: Mapping дає keys/items/get і ==, тож
порівняння з dict працює в обидва боки.
"""

from array import array
from collections.abc import Iterator, Mapping

import numpy as np

CACHE_LIMIT = 64  # скільки термінів тримати розгорнутими одночасно


class LazyPositions(Mapping[str, dict[int, list[int]]]):
    def __init__(
        self,
        buf: bytes,
        entries: dict[str, tuple["array[int]", "array[int]", int]],
        cache_limit: int = CACHE_LIMIT,
        byteswap: bool = False,
    ) -> None:
        """entries: term -> (doc_ids, tfs, зміщення блоку позицій у buf). Позиції в буфері u32 little-endian.

        byteswap=True на big-endian машині: файл little-endian, а рідний uint32 ні (як у array-шляху storage).
        """
        self._buf = buf
        self._entries = entries
        self._cache: dict[str, dict[int, list[int]]] = {}
        self._cache_limit = cache_limit
        self._byteswap = byteswap

    def __getitem__(self, term: str) -> dict[int, list[int]]:
        cached = self._cache.get(term)
        if cached is not None:
            return cached
        ids, tfs, offset = self._entries[term]  # KeyError для невідомого терміна, як у dict
        raw = np.frombuffer(self._buf, dtype=np.uint32, count=sum(tfs), offset=offset)
        if self._byteswap:
            raw = raw.byteswap()
        flat: list[int] = raw.tolist()
        per_doc: dict[int, list[int]] = {}
        start = 0
        for doc_id, tf in zip(ids, tfs, strict=True):
            per_doc[doc_id] = flat[start : start + tf]
            start += tf
        if len(self._cache) >= self._cache_limit:
            del self._cache[next(iter(self._cache))]  # найстаріший запис (dict зберігає порядок вставки)
        self._cache[term] = per_doc
        return per_doc

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries
