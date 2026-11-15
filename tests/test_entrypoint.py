"""Точка входу python -m findex (запуск __main__.py)."""
import runpy
import sys

import pytest


def test_python_dash_m_findex_prints_version(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["findex", "--version"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("findex", run_name="__main__")
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("findex")
