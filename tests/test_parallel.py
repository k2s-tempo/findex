"""Паралельна збірка: build_partial/merge, розбиття на шматки, три виконавці, CLI."""

import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner

from findex.cli import app
from findex.corpus import iter_documents, list_documents
from findex.index import DocMeta, Index, Posting, build_index, build_partial, merge
from findex.parallel import build_parallel, plan_chunks
from findex.storage import save
from findex.tokenizer import tokenize

runner = CliRunner()


def reference_index(corpus_dir: Path, with_positions: bool) -> Index:
    """Збірка з лаби 4 без змін: еталон, з яким порівнюємо нову."""
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    positions: defaultdict[str, dict[int, list[int]]] = defaultdict(dict)
    docs: list[DocMeta] = []
    for doc_id, doc in enumerate(iter_documents(corpus_dir)):
        if with_positions:
            local: defaultdict[str, list[int]] = defaultdict(list)
            for pos, token in enumerate(tokenize(doc.text)):
                local[token].append(pos)
            length = sum(len(plist) for plist in local.values())
            for term, plist in local.items():
                postings[term].append(Posting(doc_id, len(plist)))
                positions[term][doc_id] = plist
        else:
            counts = Counter(tokenize(doc.text))
            length = sum(counts.values())
            for term, tf in counts.items():
                postings[term].append(Posting(doc_id, tf))
        docs.append(DocMeta(doc_id, doc.doc_id, str(doc.path), length))
    return Index(dict(postings), docs, dict(positions) if with_positions else None)


def assert_same(actual: Index, expected: Index) -> None:
    # Index.__eq__ порівнює лише постінги (Mapping), тож решту перевіряємо явно
    assert actual.postings == expected.postings
    assert list(actual.postings) == list(expected.postings)  # і порядок термінів
    assert actual.docs == expected.docs
    assert actual.positions == expected.positions


@pytest.fixture
def big_corpus(make_corpus):
    """12 документів різного розміру, щоб було що ділити на шматки."""
    return make_corpus(
        {f"doc{i:02d}.txt": " ".join(f"w{j % (i + 2)}" for j in range(10 + i * 7)) for i in range(12)}
    )


# ---------- розбиття на шматки ----------


@pytest.mark.parametrize("n_chunks", [1, 2, 3, 5, 12, 100])
def test_plan_chunks_covers_all_paths_in_order(big_corpus, n_chunks):
    paths = list_documents(big_corpus)
    chunks = plan_chunks(paths, n_chunks)
    assert [p for _, chunk in chunks for p in chunk] == paths  # кожен шлях рівно раз, у тому ж порядку
    assert all(start == paths.index(chunk[0]) for start, chunk in chunks)  # start_id = позиція першого файлу
    assert 1 <= len(chunks) <= n_chunks


def test_plan_chunks_balances_by_file_size(tmp_path):
    sizes = [1000, 10, 10, 10, 10]
    for i, size in enumerate(sizes):
        (tmp_path / f"f{i}.txt").write_text("x" * size, encoding="utf-8")
    chunks = plan_chunks(list_documents(tmp_path), 2)
    assert [len(chunk) for _, chunk in chunks] == [1, 4]  # великий файл сам, дрібні разом


def test_plan_chunks_empty_files_split_by_count(tmp_path):
    for i in range(4):
        (tmp_path / f"f{i}.txt").write_text("", encoding="utf-8")
    assert [len(c) for _, c in plan_chunks(list_documents(tmp_path), 2)] == [2, 2]


def test_plan_chunks_edge_cases():
    assert plan_chunks([], 4) == []
    with pytest.raises(ValueError):
        plan_chunks([], 0)


# ---------- build_partial і merge ----------


@pytest.mark.parametrize("with_positions", [False, True])
def test_serial_build_matches_lab4_reference(corpus, with_positions):
    assert_same(build_index(corpus, with_positions), reference_index(corpus, with_positions))


def test_build_index_still_reports_each_document(corpus):
    seen: list[DocMeta] = []
    build_index(corpus, on_document=seen.append)
    assert [meta.doc_id for meta in seen] == [0, 1, 2]


def test_partial_doc_ids_start_at_start_id(corpus):
    paths = list_documents(corpus)
    part = build_partial(paths[1:], start_id=1, with_positions=True)
    assert [d.doc_id for d in part.docs] == [1, 2]
    assert all(p.doc_id >= 1 for plist in part.postings.values() for p in plist)


@pytest.mark.parametrize("with_positions", [False, True])
@pytest.mark.parametrize("n_chunks", [1, 2, 3, 7])
def test_merge_of_any_split_equals_serial(big_corpus, n_chunks, with_positions):
    paths = list_documents(big_corpus)
    partials = [build_partial(chunk, start, with_positions) for start, chunk in plan_chunks(paths, n_chunks)]
    assert_same(
        merge(reversed(partials)), reference_index(big_corpus, with_positions)
    )  # порядок partials неважливий


def test_merge_rejects_gap_in_doc_ids(big_corpus):
    paths = list_documents(big_corpus)
    first, third = build_partial(paths[:3], 0), build_partial(paths[6:9], 6)  # документи 3-5 відсутні
    with pytest.raises(ValueError, match="суцільними"):
        merge([first, third])


def test_merge_rejects_mixed_position_modes(corpus):
    paths = list_documents(corpus)
    with pytest.raises(ValueError, match="позиц"):
        merge([build_partial(paths[:1], 0, True), build_partial(paths[1:], 1, False)])


def test_merge_of_nothing_is_empty_index():
    index = merge([])
    assert len(index) == 0 and index.docs == [] and index.positions is None


@settings(max_examples=40, deadline=None)
@given(
    texts=st.lists(st.text(alphabet="ab c.\n", max_size=40), min_size=1, max_size=8),
    n_chunks=st.integers(min_value=1, max_value=10),
    with_positions=st.booleans(),
)
def test_property_any_chunking_equals_serial(texts, n_chunks, with_positions):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for i, text in enumerate(texts):
            (directory / f"d{i:02d}.txt").write_text(text, encoding="utf-8")
        paths = list_documents(directory)
        partials = [build_partial(c, s, with_positions) for s, c in plan_chunks(paths, n_chunks)]
        assert_same(merge(partials), reference_index(directory, with_positions))


# ---------- виконавці ----------


@pytest.mark.parametrize("executor", ["serial", "threads"])
def test_executors_give_byte_identical_files(big_corpus, tmp_path, executor):
    expected, actual = tmp_path / "expected.bin", tmp_path / "actual.bin"
    save(build_index(big_corpus, True), expected, "array")
    save(build_parallel(big_corpus, True, executor=executor, workers=3), actual, "array")
    assert actual.read_bytes() == expected.read_bytes()


@pytest.mark.slow
def test_process_executor_gives_byte_identical_file(big_corpus, tmp_path):
    expected, actual = tmp_path / "expected.bin", tmp_path / "actual.bin"
    save(build_index(big_corpus, True), expected, "array")
    save(build_parallel(big_corpus, True, executor="processes", workers=2), actual, "array")
    assert actual.read_bytes() == expected.read_bytes()


def test_on_chunk_reports_every_document_once(big_corpus):
    reported: list[int] = []
    build_parallel(big_corpus, executor="threads", workers=2, on_chunk=reported.append)
    assert sum(reported) == 12


def test_timings_are_filled(corpus):
    timings: dict[str, float] = {}
    build_parallel(corpus, executor="serial", timings=timings)
    assert set(timings) == {"map", "merge", "worker_cpu"} and all(v >= 0 for v in timings.values())


def test_invalid_arguments(corpus):
    with pytest.raises(ValueError, match="workers"):
        build_parallel(corpus, executor="threads", workers=0)
    with pytest.raises(ValueError, match="виконавець"):
        build_parallel(corpus, executor="gevent", workers=2)  # type: ignore[arg-type]  # навмисно неправильне значення


@pytest.mark.parametrize(
    "executor",
    ["serial", "threads", pytest.param("processes", marks=pytest.mark.slow)],
)
def test_worker_failure_fails_the_build(corpus, executor):
    (corpus / "zz_bad.txt").write_bytes(b"\xff\xfe\x00")  # не UTF-8
    with pytest.raises(UnicodeDecodeError):
        build_parallel(corpus, executor=executor, workers=2)


# ---------- CLI ----------


def test_cli_threads_gives_same_file_as_serial(corpus, tmp_path):
    serial, threaded = tmp_path / "serial.bin", tmp_path / "threads.bin"
    assert runner.invoke(app, ["index", str(corpus), "-o", str(serial)]).exit_code == 0
    result = runner.invoke(app, ["index", str(corpus), "-o", str(threaded), "-x", "threads", "-w", "2"])
    assert result.exit_code == 0
    assert threaded.read_bytes() == serial.read_bytes()


@pytest.mark.slow
def test_cli_processes_gives_same_file_as_serial(corpus, tmp_path):
    serial, parallel = tmp_path / "serial.bin", tmp_path / "processes.bin"
    assert runner.invoke(app, ["index", str(corpus), "-o", str(serial)]).exit_code == 0
    result = runner.invoke(app, ["index", str(corpus), "-o", str(parallel), "-x", "processes", "-w", "2"])
    assert result.exit_code == 0
    assert parallel.read_bytes() == serial.read_bytes()


def test_cli_rejects_zero_workers(corpus, tmp_path):
    result = runner.invoke(app, ["index", str(corpus), "-o", str(tmp_path / "i.bin"), "-w", "0"])
    assert result.exit_code == 2


def test_cli_rejects_unknown_executor(corpus, tmp_path):
    result = runner.invoke(app, ["index", str(corpus), "-o", str(tmp_path / "i.bin"), "-x", "gevent"])
    assert result.exit_code == 2


def test_cli_unreadable_corpus_is_one_line_error(corpus, tmp_path):
    (corpus / "zz_bad.txt").write_bytes(b"\xff\xfe\x00")
    result = runner.invoke(
        app, ["index", str(corpus), "-o", str(tmp_path / "i.bin"), "-x", "threads", "-w", "2"]
    )
    assert result.exit_code == 1
    assert "Помилка" in result.stderr and "Traceback" not in result.stderr
