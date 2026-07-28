import os

from modelscope.hub.snapshot_download import snapshot_download

# 从环境变量读取模型缓存目录，默认使用相对路径
cache_dir = os.getenv("MODELSCOPE_CACHE", "./ai_models/modelscope_cache/models")

# 下载模型到缓存目录下的 bge-m3 文件夹
model_dir = snapshot_download("BAAI/bge-m3", cache_dir=cache_dir)
print(f"模型已下载到: {model_dir}")
