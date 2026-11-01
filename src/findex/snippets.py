"""
Сніпети: фрагмент документа навколо найкращого входження термінів запиту,
з підсвіткою знайдених термінів.

Алгоритм:
  1. читаємо файл і ріжемо шапку Gutenberg (_strip_boilerplate) так само,
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
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import _strip_boilerplate
from findex.tokenizer import _TOKEN_RE

DEFAULT_RADIUS = 80
DEFAULT_MARKS = ("**", "**")


@dataclass(frozen=True)
class Snippet:
    text: str       # готовий рядок з маркерами підсвітки
    matched: tuple  # які терміни запиту реально потрапили у вікно


def prepare_text(raw: str):
    """(текст для показу, текст для пошуку). Довжини гарантовано однакові."""
    normalized = unicodedata.normalize("NFKC", raw)
    lowered = normalized.lower()
    if len(lowered) != len(normalized):   # рідкісні символи, чий lower() змінює довжину
        normalized = lowered
    return normalized, lowered


def find_matches(lowered: str, terms) -> list:
    """[(start, end, term)] для токенів тексту, що є серед термінів запиту."""
    terms = set(terms)
    return [(m.start(), m.end(), m.group())
            for m in _TOKEN_RE.finditer(lowered) if m.group() in terms]


def best_match(matches, weights, radius: int = DEFAULT_RADIUS) -> int:
    """Індекс найкращого входження за один прохід (два вказівники, O(m))."""
    n = len(matches)
    counts = Counter()
    left = right = 0
    best_i, best_key = 0, None
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


def render(text: str, matches, best: int, radius: int = DEFAULT_RADIUS,
           marks=DEFAULT_MARKS) -> Snippet:
    s, e, _ = matches[best]
    start, end = max(0, s - radius), min(len(text), e + radius)

    if start > 0:                              # не розрізати слово зліва
        ws = re.compile(r"\s").search(text, start, s)
        if ws:
            start = ws.end()
    if end < len(text):                        # ... і справа
        chunk = text[e:end]
        cut = max(chunk.rfind(" "), chunk.rfind("\n"), chunk.rfind("\t"))
        if cut >= 0:
            end = e + cut

    pieces, pos, found = [], start, set()
    for ms, me, term in matches:
        if ms >= start and me <= end:
            pieces.append(text[pos:ms])
            pieces.append(marks[0] + text[ms:me] + marks[1])
            pos = me
            found.add(term)
    pieces.append(text[pos:end])

    body = " ".join("".join(pieces).split())   # переноси рядків -> пробіли
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return Snippet(f"{prefix}{body}{suffix}", tuple(sorted(found)))


def snippet_from_text(raw_text: str, terms, weights=None, radius: int = DEFAULT_RADIUS,
                      marks=DEFAULT_MARKS) -> Snippet:
    """Чиста функція (без файлів): зручна для тестів."""
    text, lowered = prepare_text(raw_text)
    terms = set(terms)
    weights = weights or {t: 1.0 for t in terms}
    matches = find_matches(lowered, terms)
    if not matches:                            # напр. запит лише з NOT: показати початок
        head = " ".join(text[:2 * radius].split())
        return Snippet(head + ("..." if len(text) > 2 * radius else ""), ())
    return render(text, matches, best_match(matches, weights, radius), radius, marks)


def make_snippet(index, doc_id: int, terms, radius: int = DEFAULT_RADIUS,
                 marks=DEFAULT_MARKS) -> Snippet:
    """Сніпет документа doc_id з індексу для термінів запиту."""
    meta = index.docs[doc_id]
    try:
        raw = _strip_boilerplate(Path(meta.path).read_text(encoding="utf-8"))
    except OSError:
        return Snippet("(файл документа недоступний)", ())
    weights = {t: math.log(1 + index.num_docs / max(1, index.df(t))) for t in set(terms)}
    return snippet_from_text(raw, terms, weights, radius, marks)