"""
«Жадібна» версія збору статистики — для порівняння з ліниво-генераторною.

Відмінності від lazy-версії:
1. Усі документи одразу читаються в СПИСОК рядків (весь корпус у пам'яті).
2. tokenize() тут теж повертає СПИСОК (list), а не генератор —
   для кожного документа одразу будується повний список токенів.
3. Побудова "списку всіх токенів корпусу" (flat list) — 
   ще одна повна копія даних, яка не потрібна для самої задачі
   підрахунку, але типова помилка "жадібного" підходу.
"""
from pathlib import Path
from collections import Counter

from findex.corpus import _strip_boilerplate
from findex.tokenizer import _TOKEN_RE


def read_all_documents_greedy(corpus_dir: Path) -> list[str]:
    """Жадібно: читає ВСІ файли одразу і тримає їх текст у списку в пам'яті."""
    texts = []
    for path in sorted(corpus_dir.glob("*.txt")):
        raw_text = path.read_text(encoding="utf-8")
        texts.append(_strip_boilerplate(raw_text))
    return texts


def tokenize_greedy(text: str) -> list[str]:
    """Жадібно: одразу будує повний список токенів документа."""
    import unicodedata
    normalized = unicodedata.normalize("NFKC", text).lower()
    return _TOKEN_RE.findall(normalized)


def compute_stats_greedy(corpus_dir: Path):
    all_texts = read_all_documents_greedy(corpus_dir)   # весь корпус у RAM

    all_tokens = []                                      # усі токени всіх документів разом
    for text in all_texts:
        doc_tokens = tokenize_greedy(text)               # список токенів документа
        all_tokens.extend(doc_tokens)

    term_freq = Counter(all_tokens)

    return {
        "doc_count": len(all_texts),
        "token_count": len(all_tokens),
        "vocab_size": len(term_freq),
        "top_50": term_freq.most_common(50),
    }


if __name__ == "__main__":
    from findex.stats import run_with_measurement

    corpus_dir = Path("data/raw")
    result, elapsed, peak = run_with_measurement(corpus_dir, compute_stats_greedy)

    print(f"[GREEDY] Документів: {result['doc_count']}")
    print(f"[GREEDY] Токенів: {result['token_count']}")
    print(f"[GREEDY] Розмір словника: {result['vocab_size']}")
    print(f"[GREEDY] Час: {elapsed:.4f} с")
    print(f"[GREEDY] Пікова пам'ять: {peak / 1024:.1f} КБ ({peak / 1024 / 1024:.3f} МБ)")