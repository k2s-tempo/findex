"""
Оцінка якості: precision@5 для TF-IDF і BM25 на розмічених запитах.

Для кожного запиту ВРУЧНУ, ДО перегляду видачі, записано, які документи
мають бути в топ-5 (розмітка нижче). P@5 = (скільки з топ-5 є в розмітці) / 5.
Якщо розмічено менше 5 документів, максимум P@5 менший за 1 - це нормально.
"""
import argparse
from pathlib import Path

from findex.query import ranked_search
from findex.scoring import BM25, TfIdf
from findex.storage import open_index

# (запит, {імена документів, що мають бути в топ-5}) - імена як у DocMeta.name, малими літерами
LABELED = [
    ("piety OR holiness",
     {"euthyphro", "pg1656_apology", "protagoras", "laws"}),
    ("temperance",
     {"pg1580", "therepublicofplato", "laws", "gorgias", "protagoras"}),
    ("rhetoric OR orator",
     {"gorgias", "phaedrus", "menexenus"}),
    ("poets OR poetry",
     {"ion", "protagoras", "therepublicofplato", "laws", "phaedrus"}),
    ("sophist OR sophists",
     {"sophist", "protagoras", "euthydemus", "meno", "gorgias"}),
    ("pleasure",
     {"philebus", "protagoras", "gorgias", "therepublicofplato", "laws"}),
    ("wealth OR riches",
     {"eryxias", "therepublicofplato", "laws", "euthydemus"}),
    ("creation OR universe OR cosmos",
     {"timaeus", "critias", "statesman", "philebus", "laws"}),
    ("falsehood OR lie OR lies",
     {"lesserhippias", "sophist", "therepublicofplato", "laws", "cratylus"}),
    ("prison OR indictment OR trial",
     {"pg1657_crito", "pg1656_apology", "euthyphro", "gorgias", "laws"}),
]


def precision_at_k(ranked_names, relevant, k: int = 5) -> float:
    top = [n.lower() for n in ranked_names[:k]]
    return sum(1 for n in top if n in relevant) / k


def evaluate(index, labeled, scorer, k: int = 5):
    """[(запит, розмітка, імена топ-k, P@k, скільки всього знайдено)]."""
    known = {d.name.lower() for d in index.docs}
    unknown = sorted({n for _, rel in labeled for n in rel} - known)
    if unknown:   # захист від одруківок у розмітці
        raise ValueError(f"у розмітці є документи, яких нема в індексі: {unknown}")
    rows = []
    for query, relevant in labeled:
        _, total, results = ranked_search(index, query, scorer, k)
        names = [r.name for r in results]
        rows.append((query, relevant, names, precision_at_k(names, relevant, k), total))
    return rows


def _fmt(names, relevant):
    return ", ".join(n + ("✓" if n.lower() in relevant else "") for n in names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    scorers = [("BM25", BM25()), ("TF-IDF", TfIdf())]
    with open_index(args.index) as idx:
        results = {name: evaluate(idx, LABELED, s, args.k) for name, s in scorers}

    k = args.k
    print(f"| № | Запит | Розмічено | P@{k} BM25 | P@{k} TF-IDF |")
    print("|---|---|---|---|---|")
    for i, (query, rel) in enumerate(LABELED):
        bm, tf = results["BM25"][i][3], results["TF-IDF"][i][3]
        print(f"| {i + 1} | `{query}` | {len(rel)} | {bm:.2f} | {tf:.2f} |")
    means = {name: sum(r[3] for r in rows) / len(rows) for name, rows in results.items()}
    print(f"| | **Середнє** | | **{means['BM25']:.2f}** | **{means['TF-IDF']:.2f}** |")

    print("\nТоп-5 за кожним запитом (✓ = було в розмітці):\n")
    for i, (query, rel) in enumerate(LABELED):
        print(f"{i + 1}. `{query}`; розмічено: {', '.join(sorted(rel))}")
        for name, _ in scorers:
            _, _, names, p, total = results[name][i]
            print(f"   - {name}: {_fmt(names, rel)}")


if __name__ == "__main__":
    main()