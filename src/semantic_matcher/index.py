from __future__ import annotations

import json
import os
import pickle
from typing import Dict, Iterable, List, Optional, Sequence

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .config import MatcherConfig
from .text_utils import join_fields, normalize_text, tokenize


def build_document(record: Dict[str, object]) -> str:
    fields = [
        record.get("official_name", ""),
        record.get("brand", ""),
        record.get("model", ""),
        record.get("category", ""),
        record.get("description", ""),
        record.get("keywords", ""),
        record.get("domain", ""),
        record.get("year", ""),
        record.get("year_type", ""),
    ]
    return join_fields(fields)


def _write_jsonl(path: str, records: Iterable[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")


def build_index(
    catalogue_path: str,
    output_dir: str,
    model_name: Optional[str] = None,
    batch_size: int = 64,
    matcher_config: Optional[MatcherConfig] = None,
    index_version: Optional[str] = None,
) -> MatcherConfig:
    config = matcher_config or MatcherConfig()
    if model_name:
        config.model_name = model_name
    if index_version:
        config.index_version = index_version

    versioned_dir = os.path.join(output_dir, f"v{config.index_version}")
    os.makedirs(versioned_dir, exist_ok=True)

    df = pd.read_csv(catalogue_path).fillna("")
    records = df.to_dict(orient="records")

    texts: List[str] = []
    items: List[Dict[str, object]] = []
    for record in records:
        raw_text = build_document(record)
        normalized = normalize_text(raw_text)
        record = dict(record)
        record["normalized_text"] = normalized
        items.append(record)
        texts.append(normalized)

    model = SentenceTransformer(config.model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(versioned_dir, "index.faiss"))
    np.save(os.path.join(versioned_dir, "embeddings.npy"), embeddings)

    tokenized_corpus = [tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(os.path.join(versioned_dir, "bm25.pkl"), "wb") as handle:
        pickle.dump(bm25, handle)

    item_ids = [record.get("item_id", "") for record in items]
    with open(os.path.join(versioned_dir, "item_ids.json"), "w", encoding="utf-8") as handle:
        json.dump(item_ids, handle, indent=2)

    _write_jsonl(os.path.join(versioned_dir, "items.jsonl"), items)
    config.to_json(os.path.join(versioned_dir, "config.json"))

    meta = {"num_items": len(items), "embedding_dim": dim}
    with open(os.path.join(versioned_dir, "index_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    return config
