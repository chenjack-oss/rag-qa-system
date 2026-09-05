# RAG QA System

[![CI](https://github.com/chenjack-oss/rag-qa-system/actions/workflows/ci.yml/badge.svg)](https://github.com/chenjack-oss/rag-qa-system/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![中文](https://img.shields.io/badge/README-中文-red.svg)](./README.md)
[![English](https://img.shields.io/badge/README-English-blue.svg)](./README.en.md)

An enterprise-grade knowledge-base QA system built on **LangGraph**, covering the full pipeline of *document ingestion → multi-path retrieval → reranking → generation*. It parses unstructured PDF/Word documents, vectorizes and indexes them, models entities into a knowledge graph, and answers questions via multi-path recall.

> 📐 See [docs/architecture.md](./docs/architecture.md) for detailed architecture diagrams (ingestion & retrieval graphs, RRF formula).

## Features

- **Document ingestion pipeline**: MinerU PDF parsing → chunking → BGE-M3 embedding (dense + sparse) → Milvus ingestion → entity recognition → Neo4j graph modeling
- **Multi-path retrieval QA**: vector search (Milvus) + HyDE search + knowledge-graph retrieval (Neo4j), fused with weighted **RRF**, reranked by **BGE-reranker-large**, answered by **Qwen** LLM
- **Engineering design**: config/code separation (`app/conf/`), unified storage clients (`app/clients/`), LangGraph orchestration
- **Quality**: unit tests for core algorithms (`tests/`, 38 cases) + GitHub Actions CI (pytest + ruff)
- **Observability**: leveled logging (console + daily-rotated files), SSE streaming progress

## Tech Stack

| Category | Choice |
|----------|--------|
| Orchestration | LangGraph |
| Vector DB | Milvus |
| Graph DB | Neo4j |
| Document DB | MongoDB |
| Object storage | MinIO |
| Embedding | BGE-M3 (local, ModelScope) |
| Reranker | BGE-reranker-large |
| LLM | Qwen series (Alibaba DashScope, OpenAI-compatible API) |
| Doc parsing | MinerU |

## Quick Start

### 1. Install dependencies

Recommended: [uv](https://github.com/astral-sh/uv)

```bash
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
```

Fill in: `OPENAI_API_KEY` / `OPENAI_BASE_URL` (DashScope), `MILVUS_URL`, `NEO4J_URI/USERNAME/PASSWORD`, `MONGO_URL`, `MINIO_*`, model paths (`BGE_M3_PATH`, `BGE_RERANKER_LARGE`), and MinerU API credentials.

### 3. Download models

```bash
python app/tool/download_bgem3.py
python app/tool/download_reranker.py
```

### 4. Start the service

```bash
python -m app.import_process.api.file_import_service
```

### 5. Run tests

```bash
pytest tests/ -v
```

## Required Services

- Milvus 2.x (vector store)
- Neo4j 5.x (graph DB, community edition is fine)
- MongoDB 6.x (document store)
- MinIO (object storage)

## License

MIT
