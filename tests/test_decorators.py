import logging

import pytest

from findex import index as index_mod
from findex import query as query_mod
from findex import storage as storage_mod
from findex.decorators import timed


def test_timed_returns_value_and_preserves_metadata():
    @timed
    def add(a, b=1):
        """Додає числа."""
        return a + b

    assert add(2, b=3) == 5
    assert add.__name__ == "add"
    assert add.__doc__ == "Додає числа."
    assert callable(add.__wrapped__)  # functools.wraps залишає посилання на оригінал
    assert add.__wrapped__(1) == 2  # оригінал викликається без заміру


def test_timed_logs_name_and_time(caplog):
    @timed
    def work():
        return 1

    with caplog.at_level(logging.INFO, logger="findex"):
        work()
    assert len(caplog.records) == 1
    assert "work" in caplog.text and "мс" in caplog.text


def test_timed_logs_even_on_exception(caplog):
    @timed
    def boom():
        raise RuntimeError("x")

    with caplog.at_level(logging.INFO, logger="findex"), pytest.raises(RuntimeError):
        boom()
    assert "boom" in caplog.text


def test_timed_applied_to_pipeline_functions():
    for func in (index_mod.build_index, storage_mod.load, query_mod.ranked_search):
        assert hasattr(func, "__wrapped__"), func.__name__
