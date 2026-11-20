import json
import logging

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.index import build_index
from findex.storage import save

runner = CliRunner()


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("index", "search", "stats"):
        assert command in result.stdout


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and result.stdout.startswith("findex")


def test_index_creates_file(corpus, tmp_path):
    out = tmp_path / "idx.bin"
    result = runner.invoke(app, ["index", str(corpus), "-o", str(out)])
    assert result.exit_code == 0
    assert out.is_file() and "Збережено" in result.stdout and "3 док." in result.stdout


def test_index_missing_corpus_is_one_line_error(tmp_path):
    result = runner.invoke(app, ["index", str(tmp_path / "nope")])
    assert result.exit_code == 1
    assert "Помилка: каталог корпусу не знайдено" in result.stderr
    assert "Traceback" not in result.output


def test_index_empty_corpus(tmp_path):
    result = runner.invoke(app, ["index", str(tmp_path)])
    assert result.exit_code == 1 and "немає жодного .txt" in result.stderr


def test_index_unwritable_output_is_one_line_error(corpus, tmp_path):
    result = runner.invoke(app, ["index", str(corpus), "-o", str(tmp_path / "no_such_dir" / "i.bin")])
    assert result.exit_code == 1 and "не вдалося записати" in result.stderr


def test_index_pickle_format(corpus, tmp_path):
    out = tmp_path / "idx.pkl"
    assert runner.invoke(app, ["index", str(corpus), "-o", str(out), "-f", "pickle"]).exit_code == 0
    assert not out.read_bytes().startswith(b"FDX")


def test_search_table(index_file):
    result = runner.invoke(app, ["search", str(index_file), "virtue", "-k", "5"])
    assert result.exit_code == 0
    assert "meno" in result.stdout and "Знайдено 1 док." in result.stdout


def test_search_without_snippets_has_no_snippet_column(index_file):
    result = runner.invoke(app, ["search", str(index_file), "virtue", "--no-snippets"])
    assert result.exit_code == 0 and "Сніпет" not in result.stdout


def test_search_escapes_rich_markup_in_snippets(index_file):
    result = runner.invoke(app, ["search", str(index_file), "justice AND crito"])
    assert result.exit_code == 0
    assert "[Aside]" in result.stdout          # дужки з тексту не з'їдені розміткою rich


def test_search_json_is_clean_stdout(index_file):
    result = runner.invoke(app, ["search", str(index_file), "justice", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)        # у stdout лише JSON: ні прогресу, ні логів
    assert payload["total"] == 2 and payload["shown"] == 2
    assert {r["doc"] for r in payload["results"]} == {"meno", "crito"}
    assert "**justice**" in payload["results"][0]["snippet"].lower()
    assert set(payload["results"][0]) == {"rank", "score", "doc", "doc_id", "tokens", "snippet"}


def test_search_json_without_snippets(index_file):
    payload = json.loads(runner.invoke(app, ["search", str(index_file), "justice", "--json", "--no-snippets"]).stdout)
    assert all(r["snippet"] == "" for r in payload["results"])


def test_search_no_results_is_not_an_error(index_file):
    result = runner.invoke(app, ["search", str(index_file), "zzzz"])
    assert result.exit_code == 0 and "Нічого не знайдено" in result.stdout


@pytest.mark.parametrize("query", ["(virtue", "virtue)", 'virtue "soul', "virtue AND", "NOT", "   "])
def test_search_bad_query_is_one_line_error(index_file, query):
    result = runner.invoke(app, ["search", str(index_file), query])
    assert result.exit_code == 1
    assert "Помилка: некоректний запит" in result.stderr
    assert "Traceback" not in result.output


def test_search_missing_index_file(tmp_path):
    result = runner.invoke(app, ["search", str(tmp_path / "nope.bin"), "x"])
    assert result.exit_code == 1 and "файл індексу не знайдено" in result.stderr


def test_search_garbage_file(tmp_path):
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"this is not an index")
    result = runner.invoke(app, ["search", str(bad), "x"])
    assert result.exit_code == 1 and "не схожий на індекс findex" in result.stderr
    assert "Traceback" not in result.output


def test_search_truncated_index_is_one_line_error(index_file, tmp_path):
    cut = tmp_path / "cut.bin"
    cut.write_bytes(index_file.read_bytes()[:60])
    result = runner.invoke(app, ["search", str(cut), "virtue"])
    assert result.exit_code == 1 and "пошкоджений" in result.stderr


def test_pickle_index_needs_explicit_flag(corpus, tmp_path):
    pkl = tmp_path / "idx.pkl"
    save(build_index(corpus, with_positions=True), pkl, "pickle")
    refused = runner.invoke(app, ["search", str(pkl), "virtue"])
    assert refused.exit_code == 1 and "--allow-pickle" in refused.stderr
    allowed = runner.invoke(app, ["search", str(pkl), "virtue", "--allow-pickle"])
    assert allowed.exit_code == 0 and "meno" in allowed.stdout


def test_phrase_requires_positions(corpus, tmp_path):
    out = tmp_path / "nopos.bin"
    assert runner.invoke(app, ["index", str(corpus), "-o", str(out), "--no-positions"]).exit_code == 0
    result = runner.invoke(app, ["search", str(out), '"the soul"'])
    assert result.exit_code == 1 and "позиці" in result.stderr


def test_phrase_search_with_positions(index_file):
    result = runner.invoke(app, ["search", str(index_file), '"the soul"', "--json"])
    assert result.exit_code == 0 and json.loads(result.stdout)["total"] == 2


@pytest.mark.parametrize("scorer", ["bm25", "tfidf"])
def test_scorer_option(index_file, scorer):
    payload = json.loads(runner.invoke(app, ["search", str(index_file), "justice", "-s", scorer, "--json"]).stdout)
    assert payload["scorer"] == scorer and payload["total"] == 2


@pytest.mark.parametrize("engine", ["merge", "set"])
def test_engine_option_gives_same_results(index_file, engine):
    query = "(justice OR poetry) NOT crito"
    payload = json.loads(runner.invoke(app, ["search", str(index_file), query, "-e", engine, "--json"]).stdout)
    assert payload["engine"] == engine
    assert {r["doc"] for r in payload["results"]} == {"meno", "ion"}


def test_invalid_options_are_usage_errors(index_file):
    assert runner.invoke(app, ["search", str(index_file), "x", "-s", "nope"]).exit_code == 2
    assert runner.invoke(app, ["search", str(index_file), "x", "-e", "nope"]).exit_code == 2
    assert runner.invoke(app, ["search", str(index_file), "x", "-k", "0"]).exit_code == 2


def test_verbosity_levels_go_to_stderr(index_file):
    quiet = runner.invoke(app, ["search", str(index_file), "virtue"])
    assert "INFO" not in quiet.stderr
    verbose = runner.invoke(app, ["-v", "search", str(index_file), "virtue"])
    assert "INFO findex: timed: ranked_search" in verbose.stderr
    assert "cache MISS" in verbose.stderr and "INFO" not in verbose.stdout
    debug = runner.invoke(app, ["-vv", "search", str(index_file), "(virtue"])
    assert debug.exit_code == 1 and "DEBUG" in debug.stderr     # -vv показує деталі помилки


def test_logging_is_reconfigured_per_invocation(index_file):
    runner.invoke(app, ["-v", "search", str(index_file), "virtue"])
    assert logging.getLogger("findex").handlers == []     # обробник прибрано після виконання


def test_stats_index(index_file):
    result = runner.invoke(app, ["stats", str(index_file), "-n", "3"])
    assert result.exit_code == 0
    assert "Документів" in result.stdout and "Позиції токенів" in result.stdout and "Топ-3" in result.stdout


def test_stats_corpus_directory(corpus):
    result = runner.invoke(app, ["stats", str(corpus)])
    assert result.exit_code == 0 and "Унікальних термінів" in result.stdout


def test_stats_empty_directory(tmp_path):
    result = runner.invoke(app, ["stats", str(tmp_path)])
    assert result.exit_code == 1 and "немає жодного .txt" in result.stderr


def test_stats_garbage_file(tmp_path):
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"nope")
    result = runner.invoke(app, ["stats", str(bad)])
    assert result.exit_code == 1 and "не схожий на індекс" in result.stderr


def test_stats_missing_path(tmp_path):
    result = runner.invoke(app, ["stats", str(tmp_path / "nope")])
    assert result.exit_code == 1 and "не знайдено" in result.stderr
