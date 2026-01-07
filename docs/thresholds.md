# Thresholds and Fallback Logic

The matcher combines semantic similarity (cosine) with lexical BM25 scores. All scores
are normalized to 0..1 before combination.

## Defaults (stored in artifacts/config.json)

- semantic_threshold: 0.55
- lexical_threshold: 0.45
- hybrid_alpha: 0.70

## Logic

1) Compute semantic scores for the top-N FAISS candidates.
2) Compute lexical BM25 scores for the top-N lexical candidates.
3) If the best semantic score is below `semantic_threshold` and the best lexical score
   is above `lexical_threshold`, return lexical-only results (fallback mode).
4) Otherwise, compute a hybrid score:

```
hybrid = (hybrid_alpha * semantic_score) + ((1 - hybrid_alpha) * lexical_score)
```

5) Map the final score through a sigmoid calibration (a, b) to get confidence.
