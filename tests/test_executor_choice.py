"""Вибір виконавця: auto, типове число воркерів, поведінка команди findex index за замовчуванням."""

import os

import pytest
from typer.testing import CliRunner

from findex import cli, parallel
from findex.cli import app
from findex.index import build_index
from findex.parallel import AUTO_MIN_BYTES, MAX_DEFAULT_WORKERS, default_workers, resolve_executor

runner = CliRunner()


def test_auto_picks_serial_for_small_corpus():
    assert resolve_executor("auto", None, AUTO_MIN_BYTES - 1) == ("serial", 1)


def test_auto_picks_processes_from_threshold():
    executor, workers = resolve_executor("auto", None, AUTO_MIN_BYTES)
    assert executor == "processes" and 1 <= workers <= MAX_DEFAULT_WORKERS


def test_explicit_choice_beats_corpus_size():
    assert resolve_executor("processes", 2, 0) == ("processes", 2)
    assert resolve_executor("threads", None, 10**9)[0] == "threads"


def test_serial_ignores_workers():
    assert resolve_executor("serial", 8, 10**9) == ("serial", 1)


@pytest.mark.parametrize(("cores", "expected"), [(64, MAX_DEFAULT_WORKERS), (2, 2), (1, 1), (None, 1)])
def test_default_workers_is_capped_by_cores_and_limit(monkeypatch, cores, expected):
    monkeypatch.setattr(os, "cpu_count", lambda: cores)
    assert default_workers() == expected


def test_cli_default_is_serial_for_tiny_corpus(corpus, tmp_path, monkeypatch):
    seen: dict[str, object] = {}

    def spy(corpus_dir, positions, **kwargs):
        seen.update(kwargs)
        return build_index(corpus_dir, positions)

    monkeypatch.setattr(cli, "build_parallel", spy)
    assert runner.invoke(app, ["index", str(corpus), "-o", str(tmp_path / "i.bin")]).exit_code == 0
    assert seen["executor"] == "serial" and seen["workers"] == 1


def test_cli_default_switches_to_processes_for_big_corpus(corpus, tmp_path, monkeypatch):
    seen: dict[str, object] = {}

    def spy(corpus_dir, positions, **kwargs):
        seen.update(kwargs)
        return build_index(corpus_dir, positions)  # справжній пул тут не потрібен

    monkeypatch.setattr(cli, "build_parallel", spy)
    monkeypatch.setattr(parallel, "AUTO_MIN_BYTES", 1)  # «великим» стає будь-який корпус
    assert runner.invoke(app, ["index", str(corpus), "-o", str(tmp_path / "i.bin")]).exit_code == 0
    workers = seen["workers"]
    assert seen["executor"] == "processes"
    assert isinstance(workers, int) and 1 <= workers <= MAX_DEFAULT_WORKERS
