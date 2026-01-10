# Project 2: NLP Semantic Matching | MLOPS Project 2026
Created By : Safwen Gharbi | CI3, 
This project builds a hybrid semantic + lexical matcher for equipment designations. It
creates a FAISS index for embeddings, a BM25 lexical fallback, and exposes a FastAPI
endpoint for top-k matches with confidence

## Quickstart

1) Install dependencies:
```
pip install -r requirements.txt
```

2) Build the index artifacts:
```
python scripts/build_index.py --catalogue catalogue.csv --output artifacts
```

3) Fit confidence calibration:
```
python scripts/fit_calibration.py --queries queries.csv --artifacts artifacts
```

4) Run the API:
```
uvicorn api.app:app --reload
```

## Docker

Build and run with Docker:

```bash
docker build -t nlp-semantic-matcher .
docker run -p 8000:8000 nlp-semantic-matcher
```

Build with custom configuration:

```bash
docker build \
  --build-arg CATALOGUE_PATH=catalogue.csv \
  --build-arg QUERIES_PATH=queries.csv \
  --build-arg ARTIFACTS_DIR=artifacts \
  --build-arg INDEX_VERSION=1.0.0 \
  --build-arg MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  -t nlp-semantic-matcher .
```

The API will be available at `http://localhost:8000` and the ui at the same URl.

## API Usage

Once running, you can test the matcher:

```bash
curl -X POST "http://localhost:8000/match" \
     -H "Content-Type: application/json" \
     -d '{"designation": "Apple MacBook Pro 14-inch", "top_k": 3}'
```

5) Evaluate:
```
python scripts/evaluate.py --queries queries.csv --artifacts artifacts --top-k 5
```

## Artifacts

Artifacts are stored in `artifacts/`:
- `index.faiss`: semantic index
- `bm25.pkl`: lexical model
- `items.jsonl`: catalogue metadata with normalized text
- `embeddings.npy`: normalized embeddings for calibration
- `config.json`: retrieval thresholds and calibration parameters
- `index_meta.json`: index size and dimension

## Rebuilding Artifacts

### For a New Catalogue

1. Use A new catalogue.csv or update `catalogue.csv` with new items
2. Set `INDEX_VERSION` to a new version
3. Build artifacts:
   ```bash
   python scripts/build_index.py \
     --catalogue catalogue.csv \
     --output artifacts \
     --index-version 1.0.1
   ```
4. Fit calibration :
   ```bash
   python scripts/fit_calibration.py \
     --queries queries.csv \
     --artifacts artifacts \
     --index-version 1.0.1
   ```
5. Update `INDEX_VERSION` environment variable or config to use the new version

### For a New Model

1. Set `MODEL_NAME` to the new model identifier
2. Set `INDEX_VERSION` to a new version
3. Build artifacts:
   ```bash
   python scripts/build_index.py \
     --catalogue catalogue.csv \
     --output artifacts \
     --index-version 1.0.1 \
     --model sentence-transformers/new-model-name
   ```
4. Fit calibration:
   ```bash
   python scripts/fit_calibration.py \
     --queries queries.csv \
     --artifacts artifacts \
     --index-version 1.0.1
   ```

### Using Docker

Rebuild artifacts in Docker:

```bash
docker build \
  --build-arg CATALOGUE_PATH=catalogue.csv \
  --build-arg QUERIES_PATH=queries.csv \
  --build-arg INDEX_VERSION=1.0.1 \
  --build-arg MODEL_NAME=sentence-transformers/new-model-name \
  -t nlp-semantic-matcher .
```

## Documentation

Thresholds and fallback logic are described in `docs/thresholds.md`.
