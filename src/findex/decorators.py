"""
@timed - замір часу виклику з записом у лог.

Це замикання: wrapper пам'ятає func із зовнішньої функції. functools.wraps
копіює в wrapper __name__, __doc__, __module__ та ставить __wrapped__, тому
декорована функція лишається впізнаваною (help, repr, інтроспекція).

Лог пишеться рівнем INFO в логер "findex"; CLI вмикає його через logging.basicConfig.
"""
import functools
import logging
import time

log = logging.getLogger("findex")


def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            # finally: час логується і тоді, коли func підняла виняток
            elapsed_ms = (time.perf_counter() - start) * 1000
            log.info("timed: %s %.3f мс", func.__qualname__, elapsed_ms)
    return wrapper