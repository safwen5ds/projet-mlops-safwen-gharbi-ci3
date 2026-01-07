from .config import MatcherConfig
from .settings import AppSettings, get_settings

__all__ = ["MatcherConfig", "SemanticMatcher", "build_index", "AppSettings", "get_settings"]


def __getattr__(name):
    if name == "build_index":
        from .index import build_index
        return build_index
    if name == "SemanticMatcher":
        from .retrieval import SemanticMatcher
        return SemanticMatcher
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
