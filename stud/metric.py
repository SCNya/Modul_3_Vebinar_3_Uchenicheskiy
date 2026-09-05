from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import pandas as pd


def _read_table(data):
    if isinstance(data, (str, Path)):
        return pd.read_csv(data)
    if isinstance(data, pd.DataFrame):
        return data.copy()
    raise TypeError("Ожидался путь к CSV-файлу или pandas.DataFrame")


def _parse_recommendations(value) -> list[str]:
    if pd.isna(value):
        return []
    return str(value).strip().split()


def validate_submission(submission, candidates=None, expected_readers: Iterable[str] | None = None) -> None:
    sub = _read_table(submission)
    required = {"reader_id", "recommendations"}
    if set(sub.columns) != required:
        raise ValueError("В файле должны быть ровно две колонки: reader_id и recommendations")
    if sub["reader_id"].duplicated().any():
        raise ValueError("Каждый reader_id должен встречаться ровно один раз")

    if expected_readers is not None:
        expected = set(map(str, expected_readers))
        actual = set(sub["reader_id"].astype(str))
        missing = expected - actual
        extra = actual - expected
        if missing or extra:
            raise ValueError(f"Неверный набор читателей. Пропущено: {len(missing)}, лишних: {len(extra)}")

    candidate_map = None
    if candidates is not None:
        cand = _read_table(candidates)
        candidate_map = cand.groupby("reader_id")["book_id"].apply(lambda x: set(map(str, x))).to_dict()

    for row in sub.itertuples(index=False):
        recs = _parse_recommendations(row.recommendations)
        if len(recs) != 5:
            raise ValueError(f"Для {row.reader_id} должно быть ровно 5 рекомендаций")
        if len(set(recs)) != 5:
            raise ValueError(f"Для {row.reader_id} книги не должны повторяться")
        if candidate_map is not None:
            allowed = candidate_map.get(row.reader_id)
            if allowed is None:
                raise ValueError(f"Для {row.reader_id} не найден список кандидатов")
            bad = [book for book in recs if book not in allowed]
            if bad:
                raise ValueError(f"Для {row.reader_id} указаны книги не из списка кандидатов: {bad}")


def ndcg_at_5(solution, submission, candidates=None) -> float:
    truth = _read_table(solution)
    sub = _read_table(submission)
    expected = truth["reader_id"].astype(str).tolist()
    validate_submission(sub, candidates=candidates, expected_readers=expected)

    truth_map = {
        str(row.reader_id): _parse_recommendations(row.recommendations)
        for row in truth.itertuples(index=False)
    }
    pred_map = {
        str(row.reader_id): _parse_recommendations(row.recommendations)
        for row in sub.itertuples(index=False)
    }

    scores = []
    for reader_id, ideal in truth_map.items():
        # Порядок в правильном ответе задаёт убывающую полезность: 5, 4, 3, 2, 1.
        relevance = {book_id: 5 - pos for pos, book_id in enumerate(ideal)}
        dcg = 0.0
        for rank, book_id in enumerate(pred_map[reader_id], start=1):
            rel = relevance.get(book_id, 0)
            dcg += (2 ** rel - 1) / math.log2(rank + 1)
        ideal_rels = [5, 4, 3, 2, 1]
        idcg = sum((2 ** rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(ideal_rels, start=1))
        scores.append(dcg / idcg)
    return float(sum(scores) / len(scores))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Локальный расчёт NDCG@5")
    parser.add_argument("solution", help="CSV с правильными ответами")
    parser.add_argument("submission", help="CSV с рекомендациями")
    parser.add_argument("--candidates", help="CSV с кандидатами для проверки формата")
    args = parser.parse_args()

    score = ndcg_at_5(args.solution, args.submission, args.candidates)
    print(f"NDCG@5: {score:.6f}")
