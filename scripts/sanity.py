"""
Три sanity-check рейтингу на реальному корпусі (вивід у вигляді таблиць для README):
  1. рідкий термін важить більше за частий;
  2. 20-те повторення терміна майже нічого не додає;
  3. короткий документ з одним входженням вищий за дуже довгий.
"""
import argparse
from pathlib import Path

from findex.ranking import rank
from findex.scoring import BM25, TfIdf
from findex.storage import open_index

CHECKED = [("TF-IDF", TfIdf()), ("TF-IDF без норм. довжини", TfIdf(norm=False)), ("BM25", BM25())]


def check_rare_vs_common(idx):
    total_tf = lambda t: sum(p.tf for p in idx[t])
    common = max((t for t in idx if t.isalpha()), key=lambda t: (idx.df(t), total_tf(t)))
    rare = min((t for t in idx if t.isalpha() and len(t) > 4 and idx.df(t) == 1), key=lambda t: t)
    avg = round(idx.avg_doc_length)
    print(f"### 1. Рідкий проти частого (tf=1, документ середньої довжини ~{avg} токенів)\n")
    print(f"| Термін | df | " + " | ".join(n for n, _ in CHECKED) + " |")
    print("|---|---|" + "---|" * len(CHECKED))
    for term in (rare, common):
        row = [f"{s.score(1, idx.df(term), avg, idx):.3f}" for _, s in CHECKED]
        print(f"| `{term}` | {idx.df(term)} | " + " | ".join(row) + " |")
    owner = idx.docs[idx[rare][0].doc_id].name
    top = rank(idx, [common, rare], BM25(), k=1)[0]
    print(f"\nЗапит `{common} {rare}`: перший за BM25 - **{top.name}** "
          f"(єдиний документ із `{rare}` це {owner}).\n")


def check_saturation(idx):
    avg = round(idx.avg_doc_length)
    df = 3
    tfs = [1, 2, 5, 10, 20, 50, 100]
    print(f"### 2. Насичення повторів (df={df}, довжина документа ~{avg})\n")
    print("| tf | " + " | ".join(n for n, _ in CHECKED) + " |")
    print("|---|" + "---|" * len(CHECKED))
    for tf in tfs:
        print(f"| {tf} | " + " | ".join(f"{s.score(tf, df, avg, idx):.3f}" for _, s in CHECKED) + " |")
    print("\nВнесок одного додаткового повтору:\n")
    print("| Перехід | " + " | ".join(n for n, _ in CHECKED) + " |")
    print("|---|" + "---|" * len(CHECKED))
    for a, b in ((1, 2), (19, 20)):
        print(f"| {a} → {b} | " + " | ".join(
            f"{s.score(b, df, avg, idx) - s.score(a, df, avg, idx):.4f}" for _, s in CHECKED) + " |")
    print()


def check_length(idx):
    docs = sorted(idx.docs, key=lambda d: d.length)
    short, long_ = docs[0], docs[-1]
    tf_by_doc = {}
    for term in idx:
        if not term.isalpha():
            continue
        tfs = {p.doc_id: p.tf for p in idx[term]}
        if tfs.get(short.doc_id) == 1 and tfs.get(long_.doc_id) == 1:
            tf_by_doc[term] = idx.df(term)
    print("### 3. Короткий проти довгого (по одному входженню)\n")
    if not tf_by_doc:
        print("Не знайшлося терміна з tf=1 в обох документах.\n")
        return
    term = min(tf_by_doc, key=lambda t: (tf_by_doc[t], t))
    df = idx.df(term)
    print(f"Термін `{term}` (df={df}) зустрічається по одному разу в **{short.name}** "
          f"({short.length} токенів) і в **{long_.name}** ({long_.length} токенів).\n")
    print("| Scorer | " + short.name + " | " + long_.name + " | Короткий вище? |")
    print("|---|---|---|---|")
    for name, s in CHECKED:
        a = s.score(1, df, short.length, idx)
        b = s.score(1, df, long_.length, idx)
        print(f"| {name} | {a:.3f} | {b:.3f} | {'так' if a > b else 'ні (рівні)' if a == b else 'ні'} |")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    args = parser.parse_args()
    with open_index(args.index) as idx:
        check_rare_vs_common(idx)
        check_saturation(idx)
        check_length(idx)


if __name__ == "__main__":
    main()
