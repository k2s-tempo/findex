"""Залежності (Depends): індекс і кешований обчислювач беруться зі стану застосунку."""

from collections.abc import Callable, Sequence

from fastapi import HTTPException, Request

from findex.index import Index
from findex.query import Node

EvaluatorFn = Callable[[Node], Sequence[int]]


def get_index(request: Request) -> Index:
    index: Index | None = getattr(request.app.state, "index", None)
    if index is None:
        raise HTTPException(status_code=503, detail="Індекс ще не завантажений")
    return index


def get_evaluator(request: Request) -> EvaluatorFn | None:
    evaluator: EvaluatorFn | None = getattr(request.app.state, "evaluator", None)
    return evaluator
