"""Конфіг сервісу: лише змінні оточення (INDEX_PATH, LOG_LEVEL, SEARCH_MODE)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    index_path: Path = Path("index.bin")
    log_level: str = "INFO"
    # sync = маршрут def (пул потоків), async = async def (ЕКСПЕРИМЕНТ для вимірювання, див. README)
    search_mode: Literal["sync", "async"] = "sync"


@lru_cache
def get_settings() -> Settings:
    return Settings()
