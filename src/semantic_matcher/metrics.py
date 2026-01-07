from __future__ import annotations

from typing import Iterable, List, Sequence


def recall_at_k(results: Sequence[Sequence[str]], truths: Sequence[str], k: int) -> float:
    if not results:
        return 0.0
    hits = 0
    for ranked, truth in zip(results, truths):
        if truth in ranked[:k]:
            hits += 1
    return hits / float(len(results))


def mean_reciprocal_rank(results: Sequence[Sequence[str]], truths: Sequence[str]) -> float:
    if not results:
        return 0.0
    total = 0.0
    for ranked, truth in zip(results, truths):
        rank = 0
        for idx, item_id in enumerate(ranked, start=1):
            if item_id == truth:
                rank = idx
                break
        if rank:
            total += 1.0 / rank
    return total / float(len(results))
