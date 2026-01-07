import argparse
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from semantic_matcher.metrics import mean_reciprocal_rank, recall_at_k
from semantic_matcher.retrieval import SemanticMatcher
from semantic_matcher.settings import get_settings


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Evaluate Recall@k and MRR.")
    parser.add_argument("--queries", default=os.getenv("QUERIES_PATH", settings.queries_path))
    parser.add_argument("--artifacts", default=os.getenv("ARTIFACTS_DIR", settings.artifacts_dir))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--index-version", default=os.getenv("INDEX_VERSION", settings.index_version))
    args = parser.parse_args()

    queries_path = settings.resolve_path(args.queries) if not os.path.isabs(args.queries) else args.queries
    artifacts_dir = settings.resolve_path(args.artifacts) if not os.path.isabs(args.artifacts) else args.artifacts

    df = pd.read_csv(queries_path)
    if args.max_queries and args.max_queries > 0:
        df = df.head(args.max_queries)

    matcher = SemanticMatcher.from_artifacts(artifacts_dir, index_version=args.index_version)
    ranked_lists = []
    truths = []
    for row in df.itertuples(index=False):
        results = matcher.match(row.query_text, top_k=args.top_k)
        ranked_lists.append([result["item_id"] for result in results])
        truths.append(row.item_id)

    recall1 = recall_at_k(ranked_lists, truths, 1)
    recall5 = recall_at_k(ranked_lists, truths, min(5, args.top_k))
    mrr = mean_reciprocal_rank(ranked_lists, truths)

    print(f"Recall@1: {recall1:.4f}")
    print(f"Recall@5: {recall5:.4f}")
    print(f"MRR: {mrr:.4f}")


if __name__ == "__main__":
    main()
