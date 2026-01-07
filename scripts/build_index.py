import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from semantic_matcher.config import MatcherConfig
from semantic_matcher.index import build_index
from semantic_matcher.settings import get_settings


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build semantic and lexical index artifacts.")
    parser.add_argument("--catalogue", default=os.getenv("CATALOGUE_PATH", settings.catalogue_path))
    parser.add_argument("--output", default=os.getenv("ARTIFACTS_DIR", settings.artifacts_dir))
    parser.add_argument("--model", default=os.getenv("MODEL_NAME"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--index-version", default=os.getenv("INDEX_VERSION", settings.index_version))
    args = parser.parse_args()

    catalogue_path = settings.resolve_path(args.catalogue) if not os.path.isabs(args.catalogue) else args.catalogue
    output_dir = settings.resolve_path(args.output) if not os.path.isabs(args.output) else args.output

    config = MatcherConfig()
    if args.model:
        config.model_name = args.model
    config.index_version = args.index_version
    
    build_index(
        catalogue_path=catalogue_path,
        output_dir=output_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        matcher_config=config,
        index_version=args.index_version,
    )


if __name__ == "__main__":
    main()
