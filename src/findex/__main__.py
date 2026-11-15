"""Запуск пакета як модуля: python -m findex ... (те саме, що команда findex)."""

from findex.cli import app

if __name__ == "__main__":
    app(prog_name="findex")
