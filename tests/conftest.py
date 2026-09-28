"""pytest 全局引导：让测试套件不依赖本地 .env

背景：`app.utils.path_util.get_project_root()` 通过向上查找 `.env` 定位项目根目录。
本地开发环境存在该文件，但 CI / 干净检出（clone 后未配置任何凭据）时并不存在，
一旦被导入即抛 FileNotFoundError，导致依赖提示词文件的测试无法运行。

`get_project_root()` 的设计里，环境变量 `PROJECT_ROOT` 是第一优先级分支，
因此这里按「仓库根 = tests/ 的上一级」注入（setdefault，不覆盖用户已有配置），
使测试在 CI 与本地行为一致。
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
