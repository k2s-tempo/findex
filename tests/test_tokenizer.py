from findex.tokenizer import tokenize


def test_basic_words():
    assert list(tokenize("Hello world")) == ["hello", "world"]


def test_lowercase():
    assert list(tokenize("HELLO World")) == ["hello", "world"]


def test_apostrophe_inside_word():
    assert list(tokenize("Don't stop")) == ["don't", "stop"]


def test_apostrophe_possessive():
    assert list(tokenize("O'Brien's book")) == ["o'brien's", "book"]


def test_hyphen_inside_word():
    assert list(tokenize("self-driving car")) == ["self-driving", "car"]


def test_numbers_are_separate_tokens():
    assert list(tokenize("Released in 1580")) == ["released", "in", "1580"]


def test_punctuation_is_dropped():
    assert list(tokenize("Wait... really?! Yes.")) == ["wait", "really", "yes"]


def test_quotes_around_word_stripped():
    # апостроф на межі слова (лапки) — не частина токена
    assert list(tokenize("He said 'stop' now")) == ["he", "said", "stop", "now"]


def test_empty_string():
    assert list(tokenize("")) == []


def test_is_generator():
    import types
    result = tokenize("test")
    assert isinstance(result, types.GeneratorType)