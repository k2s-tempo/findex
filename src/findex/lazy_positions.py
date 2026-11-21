"""
Позиції токенів, що розгортаються з буфера файлу індексу лише на вимогу (лаба 8).

Раніше load_array одразу перетворював усі позиції на списки Python: для корпусу x10 це 10,7 млн
об'єктів int, ~380 МБ і майже половина часу завантаження. Позиції потрібні лише фразам у лапках, тож
тепер вони лишаються байтами буфера, а словник {doc_id: [позиції]} для терміна будується при першому
зверненні й кешується (кеш обмежений: позиції частого терміна це сотні тисяч чисел).

Поводиться як звичайний словник term -> {doc_id: [позиції]}: Mapping дає keys/items/get і ==, тож
порівняння з dict працює в обидва боки.

Потокобезпечно: веб-сервіс шукає в пулі потоків, а вставка й витіснення в кеші складаються з кількох
кроків, між якими GIL може віддати керування іншому потоку. Тому кеш захищений замком; саме розгортання
терміна (повільна частина) відбувається поза замком, а якщо два потоки розгорнули той самий терм,
перемагає перший (setdefault), обидва отримують однаковий результат.
"""

import threading
from array import array
from collections.abc import Iterator, Mapping

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
        self._lock = threading.Lock()

    def __getitem__(self, term: str) -> dict[int, list[int]]:
        with self._lock:
            cached = self._cache.get(term)
        if cached is not None:
            return cached
        ids, tfs, offset = self._entries[term]  # KeyError для невідомого терміна, як у dict
        import numpy as np  # лише коли справді розгортаємо позиції (імпорт ~90 мс)

        raw = np.frombuffer(self._buf, dtype=np.uint32, count=sum(tfs), offset=offset)
        if self._byteswap:
            raw = raw.byteswap()
        flat: list[int] = raw.tolist()
        per_doc: dict[int, list[int]] = {}
        start = 0
        for doc_id, tf in zip(ids, tfs, strict=True):
            per_doc[doc_id] = flat[start : start + tf]
            start += tf
        with self._lock:
            per_doc = self._cache.setdefault(
                term, per_doc
            )  # якщо інший потік уже встиг, беремо його результат
            while len(self._cache) > self._cache_limit:
                del self._cache[next(iter(self._cache))]  # найстаріший запис (dict зберігає порядок вставки)
        return per_doc

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    # Замок не серіалізується, а кеш після відновлення порожній: розгорнеться знову на вимогу.
    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        del state["_lock"]
        state["_cache"] = {}
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()
