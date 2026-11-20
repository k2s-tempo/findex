"""Pydantic-моделі стоять лише на межі: вхідні параметри і вихідні відповіді.
Всередині лишаються dataclass-и з лаб 2-3."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from findex.scoring import ScorerName


class SearchQuery(BaseModel):
    """Параметри GET /search."""

    model_config = ConfigDict(extra="forbid")

    q: str = Field(
        min_length=1, max_length=500, description="Запит у мові findex (AND/OR/NOT, фрази в лапках)."
    )
    k: int = Field(default=10, ge=1, le=50, description="Результатів на сторінці.")
    scorer: ScorerName = "bm25"
    page: int = Field(default=1, ge=1, le=100, description="Номер сторінки, з 1.")


class SearchHit(BaseModel):
    rank: int
    doc_id: int
    name: str
    score: float
    tokens: int
    snippet: str


class SearchResponse(BaseModel):
    query: str
    parsed: str
    scorer: ScorerName
    page: int
    k: int
    total: int
    total_pages: int
    elapsed_ms: float
    results: list[SearchHit]


class DocResponse(BaseModel):
    doc_id: int
    name: str
    tokens: int
    text: str


class StatsResponse(BaseModel):
    docs: int
    terms: int
    postings: int
    avg_doc_length: float
    has_positions: bool


class HealthResponse(BaseModel):
    status: Literal["ok"]
    docs: int
