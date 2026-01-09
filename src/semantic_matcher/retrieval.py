from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from .calibration import calibrate_score
from .config import MatcherConfig
from .text_utils import normalize_text


@dataclass
class MatchResult:
    item_id: str
    official_name: str
    score: float
    confidence: float
    explanation: str
    record: Dict[str, object]

    def as_dict(self) -> Dict[str, object]:
        return {
            "item_id": self.item_id,
            "official_name": self.official_name,
            "score": self.score,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "record": self.record,
        }


class SemanticMatcher:
    def __init__(
        self,
        index: faiss.Index,
        items: List[Dict[str, object]],
        bm25: BM25Okapi,
        model: SentenceTransformer,
        config: MatcherConfig,
    ) -> None:
        self.index = index
        self.items = items
        self.bm25 = bm25
        self.model = model
        self.config = config
        self._item_texts = [
            str(item.get("normalized_text", "")) for item in items
        ]

    @classmethod
    def from_artifacts(
        cls,
        artifacts_dir: str,
        model_name: Optional[str] = None,
        index_version: Optional[str] = None,
    ) -> SemanticMatcher:
        base_dir = artifacts_dir

        if index_version:
            versioned_dir = os.path.join(base_dir, f"v{index_version}")
        else:
            import glob

            version_dirs = glob.glob(os.path.join(base_dir, "v*"))
            if version_dirs:
                versioned_dir = max(version_dirs, key=os.path.getmtime)
            else:
                config = MatcherConfig()
                versioned_dir = os.path.join(base_dir, f"v{config.index_version}")

        config_path = os.path.join(versioned_dir, "config.json")
        if os.path.exists(config_path):
            config = MatcherConfig.from_json(config_path)
        else:
            config = MatcherConfig()
            if index_version:
                config.index_version = index_version

        if model_name:
            config.model_name = model_name

        index = faiss.read_index(os.path.join(versioned_dir, "index.faiss"))

        with open(os.path.join(versioned_dir, "bm25.pkl"), "rb") as handle:
            bm25: BM25Okapi = pickle.load(handle)

        items = _load_items(os.path.join(versioned_dir, "items.jsonl"))
        model = SentenceTransformer(config.model_name)

        return cls(
            index=index,
            items=items,
            bm25=bm25,
            model=model,
            config=config,
        )

    def match(self, query: str, top_k: int = 5) -> List[Dict[str, object]]:
        normalized_query = normalize_text(query)
        tokens = normalized_query.split()

        embedding = self.model.encode(
            [normalized_query], normalize_embeddings=True
        )
        embedding = np.asarray(embedding, dtype="float32")

        semantic_top_n = max(top_k, self.config.semantic_top_n)
        scores, indices = self.index.search(embedding, semantic_top_n)

        semantic_pairs = [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0
        ]

        semantic_indices = [idx for idx, _ in semantic_pairs]
        semantic_norm = {
            idx: (score + 1.0) / 2.0 for idx, score in semantic_pairs
        }

        bm25_scores = self.bm25.get_scores(tokens)
        bm25_scores = np.asarray(bm25_scores, dtype="float32")

        lexical_top_n = max(top_k, self.config.lexical_top_n)
        lexical_indices = (
            np.argsort(bm25_scores)[::-1][:lexical_top_n].tolist()
        )

        max_lexical = (
            float(bm25_scores[lexical_indices[0]])
            if lexical_indices
            else 0.0
        )

        lexical_norm = {
            idx: float(bm25_scores[idx]) / max_lexical
            if max_lexical
            else 0.0
            for idx in lexical_indices
        }

        best_semantic = max(semantic_norm.values(), default=0.0)
        best_lexical = max(lexical_norm.values(), default=0.0)

        use_lexical_fallback = (
            best_semantic < self.config.semantic_threshold
            and best_lexical >= self.config.lexical_threshold
        )

        mode = "lexical_fallback" if use_lexical_fallback else "hybrid"

        candidate_indices = set(semantic_indices) | set(lexical_indices)
        results: List[MatchResult] = []

        for idx in candidate_indices:
            semantic_score = semantic_norm.get(idx, 0.0)
            lexical_score = lexical_norm.get(idx, 0.0)

            if use_lexical_fallback:
                final_score = lexical_score
            else:
                final_score = (
                    self.config.hybrid_alpha * semantic_score
                    + (1.0 - self.config.hybrid_alpha) * lexical_score
                )

            confidence = calibrate_score(
                final_score,
                self.config.calibration_a,
                self.config.calibration_b,
            )

            record = self.items[idx]
            normalized_text = str(record.get("normalized_text", ""))

            overlap = _token_overlap(tokens, normalized_text)
            explanation = f"mode={mode}; overlap={overlap or 'none'}"

            results.append(
                MatchResult(
                    item_id=str(record.get("item_id", "")),
                    official_name=str(record.get("official_name", "")),
                    score=float(final_score),
                    confidence=float(confidence),
                    explanation=explanation,
                    record=record,
                )
            )

        results.sort(key=lambda m: m.score, reverse=True)
        return [r.as_dict() for r in results[:top_k]]


def _load_items(path: str) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            items.append(json.loads(line))
    return items


def _token_overlap(
    query_tokens: List[str],
    normalized_text: str,
    limit: int = 6,
) -> str:
    if not query_tokens or not normalized_text:
        return ""

    item_tokens = set(normalized_text.split())
    overlaps: List[str] = []
    seen = set()

    for token in query_tokens:
        if token in item_tokens and token not in seen:
            overlaps.append(token)
            seen.add(token)

    if not overlaps:
        return ""

    return ", ".join(overlaps[:limit])
