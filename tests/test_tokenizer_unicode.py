"""Токенізатор: Unicode, апострофи, дефіси, цифри (parametrize)."""

import unicodedata

import pytest

from findex.tokenizer import tokenize


@pytest.mark.parametrize(
    "text, expected",
    [
        # --- кирилиця і діакритика ---
        ("Привіт, світе!", ["привіт", "світе"]),
        ("п'ять яблук", ["п'ять", "яблук"]),
        ("naïve café", ["naïve", "café"]),
        ("Straße", ["straße"]),
        # --- грецька: фінальна сигма виходить із lower() ---
        ("ΟΔΥΣΣΕΥΣ", ["οδυσσευς"]),
        ("Σόφος", ["σόφος"]),
        # --- NFKC: лігатури й повноширинні символи ---
        ("ﬁne ﬂow", ["fine", "flow"]),
        ("ＡＢＣ１２３", ["abc", "123"]),
        # --- апострофи ---
        ("don't", ["don't"]),
        ("rock'n'roll", ["rock'n'roll"]),
        ("'tis", ["tis"]),
        ("dogs'", ["dogs"]),
        ("He said 'stop' now", ["he", "said", "stop", "now"]),
        # --- дефіси ---
        ("x-ray", ["x-ray"]),
        ("well-known self-driving", ["well-known", "self-driving"]),
        ("-start end-", ["start", "end"]),
        ("--", []),
        # --- цифри ---
        ("covid19", ["covid", "19"]),
        ("2026-10-15", ["2026", "10", "15"]),
        ("٣٤", ["٣٤"]),
        # --- підкреслення й пунктуація ---
        ("snake_case", ["snake", "case"]),
        ("!!! ... ???", []),
        ("", []),
        ("   \n\t ", []),
    ],
)
def test_tokenize_cases(text, expected):
    assert list(tokenize(text)) == expected


def test_composed_and_decomposed_forms_give_same_tokens():
    composed = "caf\u00e9"  # é одним символом
    decomposed = "cafe\u0301"  # e + комбінований акут
    assert composed != decomposed
    assert list(tokenize(composed)) == list(tokenize(decomposed)) == ["caf\u00e9"]


def test_tokens_are_nfkc_normalized():
    for token in tokenize("ﬁne ＡＢＣ naïve Ὀδυσσεύς"):
        assert token == unicodedata.normalize("NFKC", token)


def test_typographic_apostrophe_splits_token():
    # Відома межа поточних правил: U+2019 не вважається апострофом усередині слова.
    # Зафіксовано як поточна поведінка (корпус Gutenberg використовує прямий апостроф).
    assert list(tokenize("don\u2019t")) == ["don", "t"]
