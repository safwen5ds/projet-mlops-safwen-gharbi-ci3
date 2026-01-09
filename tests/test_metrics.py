import pytest

from src.semantic_matcher.metrics import mean_reciprocal_rank, recall_at_k


def test_recall_at_k_perfect():
    results = [["a", "b", "c"], ["x", "y", "z"]]
    truths = ["a", "x"]
    assert recall_at_k(results, truths, k=1) == 1.0
    assert recall_at_k(results, truths, k=3) == 1.0


def test_recall_at_k_partial():
    results = [["a", "b", "c"], ["x", "y", "z"], ["m", "n", "o"]]
    truths = ["a", "x", "p"]
    assert recall_at_k(results, truths, k=1) == pytest.approx(2.0 / 3.0)
    assert recall_at_k(results, truths, k=3) == pytest.approx(2.0 / 3.0)


def test_recall_at_k_none():
    results = [["a", "b"], ["x", "y"]]
    truths = ["z", "w"]
    assert recall_at_k(results, truths, k=2) == 0.0


def test_recall_at_k_empty_results():
    assert recall_at_k([], [], k=1) == 0.0


def test_recall_at_k_k_larger_than_results():
    results = [["a", "b"], ["x"]]
    truths = ["a", "x"]
    assert recall_at_k(results, truths, k=5) == 1.0


def test_mean_reciprocal_rank_perfect():
    results = [["a", "b", "c"], ["x", "y", "z"]]
    truths = ["a", "x"]
    assert mean_reciprocal_rank(results, truths) == 1.0


def test_mean_reciprocal_rank_second_position():
    results = [["b", "a", "c"], ["x", "y", "z"]]
    truths = ["a", "x"]
    assert mean_reciprocal_rank(results, truths) == pytest.approx(
        (1.0 / 2.0 + 1.0) / 2.0
    )


def test_mean_reciprocal_rank_third_position():
    results = [["b", "c", "a"], ["x", "y", "z"]]
    truths = ["a", "x"]
    assert mean_reciprocal_rank(results, truths) == pytest.approx(
        (1.0 / 3.0 + 1.0) / 2.0
    )


def test_mean_reciprocal_rank_not_found():
    results = [["b", "c", "d"], ["x", "y", "z"]]
    truths = ["a", "x"]
    assert mean_reciprocal_rank(results, truths) == pytest.approx((0.0 + 1.0) / 2.0)


def test_mean_reciprocal_rank_empty_results():
    assert mean_reciprocal_rank([], []) == 0.0


def test_mean_reciprocal_rank_mixed():
    results = [["a", "b"], ["c", "d"], ["e", "f"]]
    truths = ["a", "d", "g"]
    expected = (1.0 / 1.0 + 1.0 / 2.0 + 0.0) / 3.0
    assert mean_reciprocal_rank(results, truths) == pytest.approx(expected)
