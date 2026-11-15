"""
tokenize(): потоковий токенізатор із Unicode-нормалізацією.

Правила токенізації (задокументовані рішення):
1. Unicode-нормалізація: NFKC - приводить сумісні варіанти символів
   до єдиної форми (лігатури, типографські варіанти).
2. Регістр: усе приводиться до нижнього (lowercase) - пошук регістронезалежний.
3. Апострофи: усередині слова (don't, O'Brien) - частина токена;
   на межі слова (лапки навколо репліки) - відкидаються.
4. Дефіси: слово з дефісом усередині (self-driving) лишається ОДНИМ токеном.
5. Цифри: числа - окремі токени, не змішуються з літерами в одному токені.
6. Решта пунктуації відкидається і не є токеном.

Токенізатор - генератор: токени віддаються по одному, без проміжного списку.
"""
import re
import unicodedata
from collections.abc import Iterator

TOKEN_RE = re.compile(
    r"[^\W\d_]+(?:['\-][^\W\d_]+)*"   # слова з апострофами/дефісами всередині
    r"|\d+",                          # або числа окремо
    re.UNICODE,
)
_TOKEN_RE = TOKEN_RE  # старе ім'я: його імпортують dev-скрипти в scripts/


def tokenize(text: str) -> Iterator[str]:
    """Видає токени тексту по одному.

    >>> list(tokenize("Don't stop self-driving in 1580!"))
    ["don't", 'stop', 'self-driving', 'in', '1580']
    """
    lowered = unicodedata.normalize("NFKC", text).lower()
    for match in TOKEN_RE.finditer(lowered):
        yield match.group()
