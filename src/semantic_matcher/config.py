from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class MatcherConfig:
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    semantic_top_n: int = 25
    lexical_top_n: int = 25
    hybrid_alpha: float = 0.7
    semantic_threshold: float = 0.55
    lexical_threshold: float = 0.45
    calibration_a: float = 8.0
    calibration_b: float = -4.0
    index_version: str = "1.0.0"

    @classmethod
    def from_json(cls, path: str) -> "MatcherConfig":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(**data)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2, sort_keys=True)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
