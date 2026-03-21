# WikiQA

RAG prototype over Wikipedia paragraphs with:
- Wiki paragraph embedding + HNSW index build
- SQLite passage store for id-based retrieval
- Flask UI for question answering

## Project Structure

- `src/retrieval/hnsw_index.py`: shared index class (`build_from_config`, `load_from_disk`, text/vector search)
- `src/retrieval/build_hnsw.py`: CLI entrypoint to build index artifacts
- `src/api/app.py`: Flask app using the built index
- `configs/build_hnsw.yaml`: index build config
- `configs/serve.yaml`: API/service config
- `configs/access.yaml`: optional Hugging Face token config (`access_token: ...`)

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Build Index

- Edit `configs/build_hnsw.yaml`
- Add `configs/access.yaml` 
- Run:

```bash
python src/retrieval/build_hnsw.py --config configs/build_hnsw.yaml
```

This writes artifacts into `output_dir` from config:
- `index.bin` (HNSW index)
- `passages.sqlite3` (passage metadata/text, keyed by `pid`)
- `manifest.json` (index/model/config metadata)

## Run API

Edit `configs/serve.yaml`, then run:

```bash
FLASK_APP=src/api/app.py flask run --host 0.0.0.0 --port 8000
```

Open:
- `http://localhost:8000`

## Config Notes

### `configs/build_hnsw.yaml`
- `dataset_path`, `dataset_name`, `split`: Hugging Face dataset source
- `output_dir`: where index artifacts are saved
- `model`: embedding model used for indexing and retrieval
- `min_paragraph_size`, `batch_size`: paragraph/filtering and batching
- `initial_index_size`, `ef_construction`, `m`, `ef_search`, `num_threads`: HNSW params

### `configs/serve.yaml`
- `index_dir`: directory containing `manifest.json` and built artifacts
- `qa_model`: generation model
- `k`: retrieval top-k
- `max_new_tokens`: output length

## Notes

- Build and serve must use compatible embedding model settings.
- If index format changes, rebuild index before serving.
