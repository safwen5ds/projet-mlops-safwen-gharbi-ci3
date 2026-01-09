import os

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

import tempfile

import pandas as pd

from src.semantic_matcher.index import build_index
from src.semantic_matcher.retrieval import SemanticMatcher


def test_semantic_matcher_integration():
    with tempfile.TemporaryDirectory() as tmpdir:
        catalogue_path = os.path.join(tmpdir, "catalogue.csv")
        artifacts_dir = os.path.join(tmpdir, "artifacts")

        df = pd.DataFrame(
            {
                "item_id": ["item1", "item2", "item3"],
                "official_name": [
                    "Apple iPhone 13",
                    "Samsung Galaxy S21",
                    "Google Pixel 6",
                ],
                "brand": ["Apple", "Samsung", "Google"],
                "model": ["iPhone 13", "Galaxy S21", "Pixel 6"],
                "category": ["Smartphone", "Smartphone", "Smartphone"],
                "description": [
                    "Latest iPhone model",
                    "Samsung flagship phone",
                    "Google flagship device",
                ],
                "keywords": ["", "", ""],
                "domain": ["", "", ""],
                "year": ["", "", ""],
                "year_type": ["", "", ""],
            }
        )
        df.to_csv(catalogue_path, index=False)

        build_index(
            catalogue_path=catalogue_path,
            output_dir=artifacts_dir,
            model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        )

        matcher = SemanticMatcher.from_artifacts(artifacts_dir)

        results = matcher.match("iPhone 13", top_k=3)

        assert len(results) <= 3
        assert len(results) > 0
        assert results[0]["item_id"] == "item1"
        assert "item_id" in results[0]
        assert "score" in results[0]
        assert "confidence" in results[0]


def test_semantic_matcher_exact_match():
    with tempfile.TemporaryDirectory() as tmpdir:
        catalogue_path = os.path.join(tmpdir, "catalogue.csv")
        artifacts_dir = os.path.join(tmpdir, "artifacts")

        df = pd.DataFrame(
            {
                "item_id": ["laptop1", "laptop2"],
                "official_name": ["MacBook Pro", "Dell XPS"],
                "brand": ["Apple", "Dell"],
                "model": ["MacBook Pro", "XPS"],
                "category": ["Laptop", "Laptop"],
                "description": ["", ""],
                "keywords": ["", ""],
                "domain": ["", ""],
                "year": ["", ""],
                "year_type": ["", ""],
            }
        )
        df.to_csv(catalogue_path, index=False)

        build_index(
            catalogue_path=catalogue_path,
            output_dir=artifacts_dir,
            model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        )

        matcher = SemanticMatcher.from_artifacts(artifacts_dir)

        results = matcher.match("MacBook Pro", top_k=1)

        assert len(results) >= 1
        assert results[0]["item_id"] == "laptop1"
