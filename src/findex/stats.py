"""
Збір статистики корпусу за ОДИН прохід, лінива (генераторна) версія.

Вимірює:
- кількість документів
- загальну кількість токенів
- розмір словника (унікальні токени)
- топ-50 найчастіших термів
- час виконання
- пікову пам'ять (tracemalloc)
"""
import time
import tracemalloc
from collections import Counter
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenizer import tokenize


def compute_stats_lazy(corpus_dir: Path):
    """
    Лінива версія: документи й токени обробляються по одному,
    ніколи не існуючи всі одразу в пам'яті як список.
    Єдина структура, що росте, — Counter (це і є "словник" статистики,
    його розмір неминучий — саме такий є результат задачі).
    """
    doc_count = 0
    token_count = 0
    term_freq = Counter()

    for doc in iter_documents(corpus_dir):     # генератор — 1 документ за раз
        doc_count += 1
        for token in tokenize(doc.text):        # генератор — 1 токен за раз
            token_count += 1
            term_freq[token] += 1

    return {
        "doc_count": doc_count,
        "token_count": token_count,
        "vocab_size": len(term_freq),
        "top_50": term_freq.most_common(50),
    }


def run_with_measurement(corpus_dir: Path, compute_fn):
    """Обгортка: запускає compute_fn, вимірює час і піковий tracemalloc."""
    tracemalloc.start()
    start_time = time.perf_counter()

    result = compute_fn(corpus_dir)

    elapsed = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return result, elapsed, peak


if __name__ == "__main__":
    corpus_dir = Path("data/raw")
    result, elapsed, peak = run_with_measurement(corpus_dir, compute_stats_lazy)

    print(f"Документів: {result['doc_count']}")
    print(f"Токенів: {result['token_count']}")
    print(f"Розмір словника: {result['vocab_size']}")
    print(f"Час: {elapsed:.4f} с")
    print(f"Пікова пам'ять: {peak / 1024:.1f} КБ ({peak / 1024 / 1024:.3f} МБ)")
    print("\nТоп-50 термів:")
    for term, freq in result["top_50"]:
        print(f"  {term}: {freq}")