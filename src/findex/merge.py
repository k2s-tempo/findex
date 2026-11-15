"""
Операції над відсортованими списками doc_id: два вказівники (merge) і множини (set).

Обидва набори робляться над однаковими вхідними даними й дають однаковий результат:
відсортований список без повторів.
"""
from collections.abc import Sequence
from typing import Literal

Engine = Literal["merge", "set"]


def merge_and(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """Перетин двох відсортованих списків за O(n+m)."""
    out: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def merge_or(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """Об'єднання двох відсортованих списків за O(n+m)."""
    out: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def merge_diff(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """Елементи a, яких немає в b (обидва списки відсортовані)."""
    out: list[int] = []
    i = j = 0
    while i < len(a):
        if j >= len(b) or a[i] < b[j]:
            out.append(a[i])
            i += 1
        elif a[i] == b[j]:
            i += 1
            j += 1
        else:
            j += 1
    return out


def set_and(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) & set(b))


def set_or(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) | set(b))


def set_diff(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) - set(b))


def intersect(a: Sequence[int], b: Sequence[int], engine: Engine) -> list[int]:
    return merge_and(a, b) if engine == "merge" else set_and(a, b)


def union(a: Sequence[int], b: Sequence[int], engine: Engine) -> list[int]:
    return merge_or(a, b) if engine == "merge" else set_or(a, b)


def difference(a: Sequence[int], b: Sequence[int], engine: Engine) -> list[int]:
    return merge_diff(a, b) if engine == "merge" else set_diff(a, b)
