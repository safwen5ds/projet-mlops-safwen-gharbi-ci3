from __future__ import annotations

import math
from typing import Iterable, Tuple


def sigmoid(value: float) -> float:
    if value >= 0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def calibrate_score(score: float, a: float, b: float) -> float:
    return sigmoid(a * score + b)


def fit_sigmoid(
    scores: Iterable[float],
    labels: Iterable[int],
    max_iter: int = 800,
    lr: float = 0.2,
    l2: float = 0.0,
) -> Tuple[float, float]:
    scores_list = list(scores)
    labels_list = list(labels)
    if not scores_list:
        return 1.0, 0.0
    a, b = 1.0, 0.0
    n = float(len(scores_list))
    for _ in range(max_iter):
        grad_a = 0.0
        grad_b = 0.0
        for score, label in zip(scores_list, labels_list):
            pred = sigmoid(a * score + b)
            diff = pred - label
            grad_a += diff * score
            grad_b += diff
        grad_a = grad_a / n + l2 * a
        grad_b = grad_b / n
        a -= lr * grad_a
        b -= lr * grad_b
    return a, b
