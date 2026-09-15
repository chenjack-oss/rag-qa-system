# RAG 知识问答服务镜像
# 依赖服务（Milvus/Neo4j/MongoDB/MinIO）由 docker-compose.yml 编排
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/huggingface_cache \
    MODELSCOPE_CACHE=/app/modelscope_cache

WORKDIR /app

# 系统依赖（Word/文档处理与编译所需的基础库）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖再拷代码，充分利用层缓存
COPY pyproject.toml uv.lock README.md ./
RUN pip install --upgrade pip && pip install uv && uv sync --frozen --no-dev

COPY app ./app
COPY prompts ./prompts
COPY evaluation ./evaluation

# 模型缓存目录挂载点
VOLUME ["/app/modelscope_cache", "/app/huggingface_cache"]

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:8001/health || exit 1

# 查询服务（导入服务可按需替换为 app.import_process.api.file_import_service）
CMD ["uv", "run", "uvicorn", "app.query_process.api.query_service:app", "--host", "0.0.0.0", "--port", "8001"]
