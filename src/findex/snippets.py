"""
Сніпети: фрагмент документа навколо найкращого входження термінів запиту,
з підсвіткою знайдених термінів.

Алгоритм:
  1. читаємо файл і ріжемо шапку Gutenberg (strip_boilerplate) так само,
     як при індексації, щоб токенізація збігалася;
  2. знаходимо символьні межі токенів тим самим регулярним виразом токенізатора
     (на NFKC-тексті, без втрати регістру для показу);
  3. «найкраще входження» - те, навколо якого (+-radius символів) сума idf
     РІЗНИХ термінів запиту найбільша (так рідкі слова переважують стоп-слова),
     далі більше входжень, далі раніше в тексті;
  4. вікно +-radius навколо нього, межі зсуваються до пробілів (слова не
     розрізаються), знайдені терміни обгортаються маркерами.
"""

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import strip_boilerplate
from findex.index import Index

DEFAULT_RADIUS = 80
DEFAULT_MARKS = ("**", "**")

Match = tuple[int, int, str]  # (початок, кінець, термін)


@dataclass(frozen=True)
class Snippet:
    text: str  # готовий рядок з маркерами підсвітки
    matched: tuple[str, ...]  # які терміни запиту реально потрапили у вікно


def prepare_text(raw: str) -> tuple[str, str]:
    """(текст для показу, текст для пошуку). Довжини гарантовано однакові."""
    normalized = unicodedata.normalize("NFKC", raw)
    lowered = normalized.lower()
    if len(lowered) != len(normalized):  # рідкісні символи, чий lower() змінює довжину
        normalized = lowered
    return normalized, lowered


_LETTER = re.compile(r"[^\W\d_]")  # той самий клас символів, що й у TOKEN_RE
_DIGIT = re.compile(r"\d")
_NUMBER = re.compile(r"\d+")
_JOINERS = "'-"  # апостроф і дефіс усередині слова не розривають токен


def _is_whole_token(text: str, start: int, end: int, numeric: bool) -> bool:
    """True, якщо text[start:end] є ПОВНИМ токеном за правилами TOKEN_RE, а не частиною довшого."""
    n = len(text)
    if numeric:
        return not (start > 0 and _DIGIT.match(text[start - 1])) and not (end < n and _DIGIT.match(text[end]))
    if start > 0:
        prev = text[start - 1]
        if _LETTER.match(prev) or (prev in _JOINERS and start > 1 and _LETTER.match(text[start - 2])):
            return False
    if end < n:
        nxt = text[end]
        if _LETTER.match(nxt) or (nxt in _JOINERS and end + 1 < n and _LETTER.match(text[end + 1])):
            return False
    return True


def find_matches(lowered: str, terms: Iterable[str]) -> list[Match]:
    """[(start, end, term)] для токенів тексту, що є серед термінів запиту.

    Не проганяємо TOKEN_RE по всьому тексту (мільйони токенів заради кількох термінів):
    шукаємо кожен термін через str.find і перевіряємо межі токена правилами токенізатора.
    Результат такий самий, як у повного проходу; терміни мають бути справжніми токенами.
    """
    found: list[Match] = []
    for term in set(terms):
        if not term:
            continue
        numeric = _NUMBER.fullmatch(term) is not None
        pos = lowered.find(term)
        while pos != -1:
            end = pos + len(term)
            if _is_whole_token(lowered, pos, end, numeric):
                found.append((pos, end, term))
            pos = lowered.find(term, pos + 1)
    found.sort()
    return found


def best_match(matches: Sequence[Match], weights: Mapping[str, float], radius: int = DEFAULT_RADIUS) -> int:
    """Індекс найкращого входження за один прохід (два вказівники, O(m))."""
    n = len(matches)
    counts: Counter[str] = Counter()
    left = right = 0
    best_i = 0
    best_key: tuple[float, int, int] | None = None
    for i, (s, e, _) in enumerate(matches):
        lo, hi = s - radius, e + radius
        while right < n and matches[right][0] <= hi:
            counts[matches[right][2]] += 1
            right += 1
        while matches[left][1] < lo:
            counts[matches[left][2]] -= 1
            left += 1
        weight = sum(weights[t] for t, c in counts.items() if c > 0)
        key = (round(weight, 9), right - left, -i)
        if best_key is None or key > best_key:
            best_key, best_i = key, i
    return best_i


def render(
    text: str,
    matches: Sequence[Match],
    best: int,
    radius: int = DEFAULT_RADIUS,
    marks: tuple[str, str] = DEFAULT_MARKS,
) -> Snippet:
    s, e, _ = matches[best]
    start, end = max(0, s - radius), min(len(text), e + radius)

    if start > 0:  # не розрізати слово зліва
        ws = re.compile(r"\s").search(text, start, s)
        if ws:
            start = ws.end()
    if end < len(text):  # ... і справа
        chunk = text[e:end]
        cut = max(chunk.rfind(" "), chunk.rfind("\n"), chunk.rfind("\t"))
        if cut >= 0:
            end = e + cut

    pieces: list[str] = []
    pos = start
    found: set[str] = set()
    for ms, me, term in matches:
        if ms >= start and me <= end:
            pieces.append(text[pos:ms])
            pieces.append(marks[0] + text[ms:me] + marks[1])
            pos = me
            found.add(term)
    pieces.append(text[pos:end])

    body = " ".join("".join(pieces).split())  # переноси рядків -> пробіли
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return Snippet(f"{prefix}{body}{suffix}", tuple(sorted(found)))


def snippet_from_text(
    raw_text: str,
    terms: Iterable[str],
    weights: Mapping[str, float] | None = None,
    radius: int = DEFAULT_RADIUS,
    marks: tuple[str, str] = DEFAULT_MARKS,
) -> Snippet:
    """Чиста функція (без файлів): зручна для тестів."""
    text, lowered = prepare_text(raw_text)
    wanted = set(terms)
    weight_map: Mapping[str, float] = weights if weights else {t: 1.0 for t in wanted}
    matches = find_matches(lowered, wanted)
    if not matches:  # напр. запит лише з NOT: показати початок
        head = " ".join(text[: 2 * radius].split())
        return Snippet(head + ("..." if len(text) > 2 * radius else ""), ())
    return render(text, matches, best_match(matches, weight_map, radius), radius, marks)


def make_snippet(
    index: Index,
    doc_id: int,
    terms: Iterable[str],
    radius: int = DEFAULT_RADIUS,
    marks: tuple[str, str] = DEFAULT_MARKS,
) -> Snippet:
    """Сніпет документа doc_id з індексу для термінів запиту."""
    meta = index.docs[doc_id]
    try:
        raw = strip_boilerplate(Path(meta.path).read_text(encoding="utf-8"))
    except OSError:
        return Snippet("(файл документа недоступний)", ())
    wanted = set(terms)
    weights = {t: math.log(1 + index.num_docs / max(1, index.df(t))) for t in wanted}
    return snippet_from_text(raw, wanted, weights, radius, marks)
