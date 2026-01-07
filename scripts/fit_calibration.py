import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from semantic_matcher.calibration import fit_sigmoid
from semantic_matcher.config import MatcherConfig
from semantic_matcher.settings import get_settings
from semantic_matcher.text_utils import normalize_text


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Fit sigmoid calibration parameters.")
    parser.add_argument("--queries", default=os.getenv("QUERIES_PATH", settings.queries_path))
    parser.add_argument("--artifacts", default=os.getenv("ARTIFACTS_DIR", settings.artifacts_dir))
    parser.add_argument("--negatives-per-query", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index-version", default=os.getenv("INDEX_VERSION", settings.index_version))
    args = parser.parse_args()

    queries_path = settings.resolve_path(args.queries) if not os.path.isabs(args.queries) else args.queries
    artifacts_dir = settings.resolve_path(args.artifacts) if not os.path.isabs(args.artifacts) else args.artifacts
    versioned_dir = os.path.join(artifacts_dir, f"v{args.index_version}")

    config_path = os.path.join(versioned_dir, "config.json")
    config = MatcherConfig.from_json(config_path)

    embeddings = np.load(os.path.join(versioned_dir, "embeddings.npy"))
    with open(os.path.join(versioned_dir, "item_ids.json"), "r", encoding="utf-8") as handle:
        item_ids = json.load(handle)
    id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}

    model = SentenceTransformer(config.model_name)
    df = pd.read_csv(queries_path)
    if args.max_queries and args.max_queries > 0:
        df = df.head(args.max_queries)

    rng = np.random.default_rng(args.seed)
    positives = []
    negatives = []
    for row in df.itertuples(index=False):
        query_text = normalize_text(row.query_text)
        query_emb = model.encode([query_text], normalize_embeddings=True)[0]
        correct_index = id_to_index.get(row.item_id)
        if correct_index is None:
            continue
        positives.append(float(np.dot(query_emb, embeddings[correct_index])))
        for _ in range(args.negatives_per_query):
            neg_index = int(rng.integers(0, len(item_ids)))
            if neg_index == correct_index:
                continue
            negatives.append(float(np.dot(query_emb, embeddings[neg_index])))

    scores = positives + negatives
    labels = [1] * len(positives) + [0] * len(negatives)
    a, b = fit_sigmoid(scores, labels)
    config.calibration_a = a
    config.calibration_b = b
    config.to_json(config_path)

    print(f"Updated calibration: a={a:.4f}, b={b:.4f}")


if __name__ == "__main__":
    main()
