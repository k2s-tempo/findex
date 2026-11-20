"""HTML-сторінки: головна з пошуком (GET /) і перегляд документа (GET /d/{doc_id})."""

import re
from pathlib import Path
from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

from findex.corpus import strip_boilerplate
from findex.index import Index
from findex.scoring import ScorerName
from findex.web.deps import EvaluatorFn, get_evaluator, get_index
from findex.web.service import PAGE_MARKS, SearchPage, search

WEB_DIR = Path(__file__).parent
PAGE_SIZE = 10
EXAMPLES = ["justice AND virtue", "courage OR temperance", '"the soul"', "wisdom NOT sophist"]

templates = Jinja2Templates(directory=WEB_DIR / "templates")

IndexDep = Annotated[Index, Depends(get_index)]
EvaluatorDep = Annotated[EvaluatorFn | None, Depends(get_evaluator)]


def highlight(text: str) -> Markup:
    """Екранує текст, а службові маркери сніпета перетворює на <mark>."""
    safe = str(escape(text)).replace(PAGE_MARKS[0], "<mark>").replace(PAGE_MARKS[1], "</mark>")
    return Markup(safe)


templates.env.filters["hl"] = highlight


def _qs(q: str, scorer: ScorerName, page: int = 1) -> str:
    return urlencode({"q": q, "scorer": scorer, "page": page})


def make_pages_router() -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    def home(
        request: Request,
        index: IndexDep,
        evaluator: EvaluatorDep,
        q: Annotated[str, Query(max_length=500)] = "",
        scorer: ScorerName = "bm25",
        page: Annotated[int, Query(ge=1, le=100)] = 1,
    ) -> HTMLResponse:
        result: SearchPage | None = None
        error: str | None = None
        if q.strip():
            try:
                result = search(index, evaluator, q, scorer, PAGE_SIZE, page, marks=PAGE_MARKS)
            except ValueError as e:
                error = f"Некоректний запит: {e}"
        has_prev = result is not None and page > 1
        has_next = result is not None and page < result.total_pages
        context: dict[str, object] = {
            "q": q,
            "scorer": scorer,
            "page": page,
            "result": result,
            "error": error,
            "examples": EXAMPLES,
            "docs": index.num_docs,
            "terms": len(index),
            "prev_url": "/?" + _qs(q, scorer, page - 1) if has_prev else None,
            "next_url": "/?" + _qs(q, scorer, page + 1) if has_next else None,
            "view_qs": _qs(q, scorer, page),
        }
        status = 422 if error else 200
        return templates.TemplateResponse(request, "index.html", context, status_code=status)

    @router.get("/d/{doc_id}", response_class=HTMLResponse)
    def view_doc(
        request: Request,
        doc_id: int,
        index: IndexDep,
        q: Annotated[str, Query(max_length=500)] = "",
        scorer: ScorerName = "bm25",
        page: Annotated[int, Query(ge=1, le=100)] = 1,
    ) -> HTMLResponse:
        if not 0 <= doc_id < index.num_docs:
            raise HTTPException(status_code=404, detail=f"Документ {doc_id} не знайдено")
        meta = index.docs[doc_id]
        try:
            text = strip_boilerplate(Path(meta.path).read_text(encoding="utf-8"))
        except OSError as e:
            raise HTTPException(status_code=404, detail="Текст документа недоступний") from e
        paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]
        context: dict[str, object] = {
            "name": meta.name,
            "tokens": meta.length,
            "paragraphs": paragraphs,
            "back_url": "/?" + _qs(q, scorer, page) if q.strip() else "/",
            "docs": index.num_docs,
            "terms": len(index),
        }
        return templates.TemplateResponse(request, "doc.html", context)

    return router
