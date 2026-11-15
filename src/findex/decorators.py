"""
@timed - замір часу виклику з записом у лог.

Це замикання: wrapper пам'ятає func із зовнішньої функції. functools.wraps
копіює в wrapper __name__, __doc__, __module__ та ставить __wrapped__, а
ParamSpec зберігає сигнатуру оригіналу для перевірки типів.

Лог пишеться рівнем INFO в логер "findex"; CLI вмикає його через -v.
"""

import functools
import logging
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

log = logging.getLogger("findex")


def timed(func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            # finally: час логується і тоді, коли func підняла виняток
            elapsed_ms = (time.perf_counter() - start) * 1000
            log.info("timed: %s %.3f мс", func.__qualname__, elapsed_ms)

    return wrapper
