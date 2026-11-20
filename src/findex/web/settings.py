"""Конфіг сервісу: лише змінні оточення (INDEX_PATH, LOG_LEVEL, SEARCH_MODE, MAX_CONCURRENT_SEARCHES)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    index_path: Path = Path("index.bin")
    log_level: str = "INFO"
    # sync = маршрут def (пул потоків), async = async def (ЕКСПЕРИМЕНТ для вимірювання, див. README)
    search_mode: Literal["sync", "async"] = "sync"
    # Скільки пошуків одночасно виконують важку роботу (сніпети читають і копіюють тексти до ~1,5 МБ).
    # Більше немає сенсу: під GIL вони все одно йдуть по черзі, а пам'ять росте з кожним.
    max_concurrent_searches: int = Field(default=4, ge=1, le=64)


@lru_cache
def get_settings() -> Settings:
    return Settings()
