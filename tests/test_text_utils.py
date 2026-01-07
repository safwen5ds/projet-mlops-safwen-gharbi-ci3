import pytest
from src.semantic_matcher.text_utils import normalize_text, tokenize


def test_normalize_text_basic():
    assert normalize_text("Hello World") == "hello world"


def test_normalize_text_with_accents():
    assert normalize_text("Café") == "cafe"
    assert normalize_text("naïve") == "naive"
    assert normalize_text("résumé") == "resume"


def test_normalize_text_removes_special_chars():
    assert normalize_text("Hello, World!") == "hello world"
    assert normalize_text("test@example.com") == "test example com"
    assert normalize_text("item-123") == "item 123"


def test_normalize_text_lowercase():
    assert normalize_text("HELLO WORLD") == "hello world"
    assert normalize_text("MiXeD CaSe") == "mixed case"


def test_normalize_text_removes_noise_tokens():
    assert normalize_text("buy price") == ""
    assert normalize_text("hello price world") == "hello world"
    assert normalize_text("acheter prix") == ""


def test_normalize_text_none():
    assert normalize_text(None) == ""


def test_normalize_text_empty():
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""


def test_tokenize_basic():
    assert tokenize("Hello World") == ["hello", "world"]


def test_tokenize_with_special_chars():
    assert tokenize("Hello, World!") == ["hello", "world"]


def test_tokenize_removes_noise():
    assert tokenize("buy price") == []
    assert tokenize("hello price world") == ["hello", "world"]


def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize("   ") == []

