import pytest

from src.semantic_matcher.calibration import fit_sigmoid, sigmoid


def test_sigmoid_zero():
    assert sigmoid(0.0) == pytest.approx(0.5)


def test_sigmoid_positive():
    assert sigmoid(1.0) > 0.5
    assert sigmoid(10.0) > 0.9


def test_sigmoid_negative():
    assert sigmoid(-1.0) < 0.5
    assert sigmoid(-10.0) < 0.1


def test_fit_sigmoid_empty():
    a, b = fit_sigmoid([], [])
    assert a == 1.0
    assert b == 0.0


def test_fit_sigmoid_perfect_separation():
    scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    labels = [1, 1, 1, 0, 0, 0]
    a, b = fit_sigmoid(scores, labels, max_iter=100)
    assert isinstance(a, float)
    assert isinstance(b, float)


def test_fit_sigmoid_deterministic():
    scores = [0.9, 0.8, 0.3, 0.2]
    labels = [1, 1, 0, 0]
    a1, b1 = fit_sigmoid(scores, labels, max_iter=50, lr=0.1)
    a2, b2 = fit_sigmoid(scores, labels, max_iter=50, lr=0.1)
    assert a1 == pytest.approx(a2)
    assert b1 == pytest.approx(b2)


def test_fit_sigmoid_all_ones():
    scores = [0.5, 0.6, 0.7]
    labels = [1, 1, 1]
    a, b = fit_sigmoid(scores, labels, max_iter=100)
    assert isinstance(a, float)
    assert isinstance(b, float)


def test_fit_sigmoid_all_zeros():
    scores = [0.3, 0.4, 0.5]
    labels = [0, 0, 0]
    a, b = fit_sigmoid(scores, labels, max_iter=100)
    assert isinstance(a, float)
    assert isinstance(b, float)


def test_fit_sigmoid_small_example():
    scores = [0.8, 0.2]
    labels = [1, 0]
    a, b = fit_sigmoid(scores, labels, max_iter=200)
    assert a > 0
    pred_high = sigmoid(a * 0.8 + b)
    pred_low = sigmoid(a * 0.2 + b)
    assert pred_high > pred_low
