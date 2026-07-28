import os

from modelscope.hub.snapshot_download import snapshot_download

# 从环境变量读取模型缓存目录，默认使用相对路径
cache_dir = os.getenv("MODELSCOPE_CACHE", "./ai_models/modelscope_cache/models")
local_dir = os.path.join(cache_dir, "rerank")

snapshot_download(
    model_id="BAAI/bge-reranker-large",
    cache_dir=local_dir,
)

print("下载完成，模型目录：", local_dir)
