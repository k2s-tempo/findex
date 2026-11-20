"""Маршрути API: /search, /docs/{doc_id}, /stats, /health."""

import logging
from pathlib import Path as FsPath
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from findex.corpus import strip_boilerplate
from findex.index import Index
from findex.web.deps import EvaluatorFn, get_evaluator, get_index
from findex.web.models import (
    DocResponse,
    HealthResponse,
    SearchHit,
    SearchQuery,
    SearchResponse,
    StatsResponse,
)
from findex.web.service import search

log = logging.getLogger("findex.web")

IndexDep = Annotated[Index, Depends(get_index)]
EvaluatorDep = Annotated[EvaluatorFn | None, Depends(get_evaluator)]
QueryDep = Annotated[SearchQuery, Query()]


def _search_response(params: SearchQuery, index: Index, evaluator: EvaluatorFn | None) -> SearchResponse:
    try:
        result = search(index, evaluator, params.q, params.scorer, params.k, params.page)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Некоректний запит: {e}") from e
    return SearchResponse(
        query=params.q,
        parsed=result.parsed,
        scorer=params.scorer,
        page=params.page,
        k=params.k,
        total=result.total,
        total_pages=result.total_pages,
        elapsed_ms=round(result.elapsed_ms, 3),
        results=[SearchHit(**vars(h)) for h in result.hits],
    )


def make_router(search_mode: str) -> APIRouter:
    router = APIRouter()

    if search_mode == "async":

        @router.get("/search", response_model=SearchResponse)
        async def search_async(params: QueryDep, index: IndexDep, evaluator: EvaluatorDep) -> SearchResponse:
            # ЕКСПЕРИМЕНТ: уся CPU-робота виконується прямо в циклі подій і блокує його.
            return _search_response(params, index, evaluator)

    else:

        @router.get("/search", response_model=SearchResponse)
        def search_sync(params: QueryDep, index: IndexDep, evaluator: EvaluatorDep) -> SearchResponse:
            # звичайний def: FastAPI виконує його в пулі потоків, цикл подій вільний
            return _search_response(params, index, evaluator)

    @router.get("/docs/{doc_id}", response_model=DocResponse)
    def get_doc(doc_id: int, index: IndexDep) -> DocResponse:
        if not 0 <= doc_id < index.num_docs:
            raise HTTPException(status_code=404, detail=f"Документ {doc_id} не знайдено")
        meta = index.docs[doc_id]
        try:
            text = strip_boilerplate(FsPath(meta.path).read_text(encoding="utf-8"))
        except OSError as e:
            log.warning("файл документа %d недоступний: %s", doc_id, e)
            raise HTTPException(status_code=404, detail="Текст документа недоступний") from e
        return DocResponse(doc_id=doc_id, name=meta.name, tokens=meta.length, text=text)

    @router.get("/stats", response_model=StatsResponse)
    def stats(index: IndexDep) -> StatsResponse:
        return StatsResponse(
            docs=index.num_docs,
            terms=len(index),
            postings=sum(len(p) for p in index.postings.values()),
            avg_doc_length=round(index.avg_doc_length, 2),
            has_positions=index.has_positions,
        )

    @router.get("/health", response_model=HealthResponse)
    async def health(index: IndexDep) -> HealthResponse:
        return HealthResponse(status="ok", docs=index.num_docs)

    return router
