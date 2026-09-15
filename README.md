# RAG知识问答系统

[![CI](https://github.com/chenjack-oss/rag-qa-system/actions/workflows/ci.yml/badge.svg)](https://github.com/chenjack-oss/rag-qa-system/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![中文](https://img.shields.io/badge/README-中文-red.svg)](./README.md)
[![English](https://img.shields.io/badge/README-English-blue.svg)](./README.en.md)

基于 LangGraph 构建的企业级知识库系统，覆盖「文档入库 → 多路检索 → 重排 → 生成」完整链路，支持 PDF/Word 等非结构化文档的解析、向量化、图谱建模与多路召回问答。

> 📐 架构详解见 [docs/architecture.md](./docs/architecture.md)（含导入/检索双链路流程图与 RRF 公式）

## 项目特性

- **文档导入流水线**：MinerU 解析 PDF → 文档切片 → BGE-M3 向量化 → Milvus 入库 → 实体识别 → Neo4j 图谱建模
- **多路检索问答**：向量检索（Milvus）+ HyDE 检索 + 图谱检索（Neo4j）多路召回，带权重 RRF 融合，BGE-reranker-large 重排序，Qwen 大模型生成
- **量化评测体系**：检索指标（hit@k / MRR / nDCG）+ 消融实验框架，线上旁路采集检索结果，离线复现评测，用数据说话验证每个组件的贡献
- **A/B 实验能力**：确定性哈希分流 + MongoDB 指标记录 + 分组分析，支撑精排/召回策略的线上对照实验与灰度发布
- **工程化设计**：配置与代码分离（`app/conf/`）、客户端统一封装（`app/clients/`）、LangGraph 编排导入流程、Docker Compose 一键部署全套服务
- **质量保障**：核心算法与 A/B 框架单元测试（`tests/`，97 个用例）+ GitHub Actions CI（测试 + ruff lint）
- **可观测**：内置分级日志（控制台 + 文件，按天滚动）、SSE 流式进度推送

## 技术栈

| 类别 | 选型 |
|------|------|
| 编排框架 | LangGraph |
| 向量库 | Milvus |
| 图数据库 | Neo4j |
| 文档库 | MongoDB |
| 对象存储 | MinIO |
| Embedding | BGE-M3（魔搭本地部署） |
| 重排序 | BGE-reranker-large |
| LLM | Qwen 系列（阿里云百炼 DashScope） |
| 文档解析 | MinerU |

## 目录结构

```
.
├── app/
│   ├── clients/          # 存储客户端封装（Milvus/MinIO/MongoDB/Neo4j）
│   ├── conf/             # 配置模块（embedding/lm/milvus/mineru/minio/reranker）
│   ├── core/             # 核心工具（prompt 加载、日志）
│   ├── experiments/      # A/B 实验框架（确定性分流 + 指标记录 + 分组分析）
│   ├── import_process/   # 文档导入流程
│   │   ├── agent/        # LangGraph 导入图（nodes + state）
│   │   └── api/          # 文件导入服务
│   ├── lm/               # 大模型工具（embedding/reranker/lm 调用）
│   ├── query_process/    # 查询问答流程
│   ├── tool/             # 模型下载等辅助工具
│   └── utils/            # 通用工具（含评测采集/消融开关旁路）
├── evaluation/           # 评测体系（指标库 + 离线评测 + 消融实验 + 样例报告）
├── docs/                 # 架构文档
├── prompts/              # 提示词模板
├── test/                 # 本地调试脚本
├── tests/                # 单元测试（pytest，CI 自动执行）
├── .github/workflows/    # GitHub Actions CI
├── Dockerfile            # 服务镜像
├── docker-compose.yml    # 一键编排（Milvus/Neo4j/MongoDB/MinIO + 双服务）
├── pyproject.toml
├── uv.lock
└── .env.example          # 环境变量模板
```

## 快速开始

### 1. 安装依赖

推荐使用 [uv](https://github.com/astral-sh/uv) 管理依赖：

```bash
uv sync
```

或使用 pip：

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

按需修改 `.env` 中的以下配置：

- **LLM**：`OPENAI_API_KEY`、`OPENAI_BASE_URL`（阿里云百炼 DashScope 兼容 OpenAI 接口）
- **Milvus**：`MILVUS_URL`
- **Neo4j**：`NEO4J_URI`、`NEO4J_USERNAME`、`NEO4J_PASSWORD`
- **MongoDB**：`MONGO_URL`
- **MinIO**：`MINIO_ENDPOINT`、`MINIO_ACCESS_KEY`、`MINIO_SECRET_KEY`
- **Embedding 模型路径**：`MODELSCOPE_CACHE`、`BGE_M3_PATH`、`BGE_RERANKER_LARGE`
- **MinerU**：`MINERU_API_TOKEN`、`MINERU_BASE_URL`

### 3. 下载模型

首次运行需下载 BGE-M3 与 reranker 模型：

```bash
python app/tool/download_bgem3.py
python app/tool/download_reranker.py
```

### 4. 启动服务（两种方式）

**方式一：本地启动**

```bash
python -m app.query_process.api.query_service     # 查询服务 :8001
python -m app.import_process.api.file_import_service  # 导入服务 :8002
```

**方式二：Docker Compose 一键部署**（推荐，含全部依赖服务）

```bash
cp .env.example .env   # 修改其中的密钥与账号
docker compose up -d   # Milvus + Neo4j + MongoDB + MinIO + 双服务
docker compose logs -f query-service   # 观察启动状态
```

服务就绪后访问 `http://localhost:8001/health` 确认健康，`http://localhost:8001/chat.html` 进入对话页面。

### 5. 运行单元测试

```bash
pytest tests/ -v
```

测试覆盖核心纯函数算法：RRF 多路融合、Milvus 表达式转义、稀疏向量 L2 归一化、JSON 格式化、动态 TopK 截断、A/B 确定性分流。CI 会在每次 push 时自动执行。

## 量化评测与消融实验

检索质量的"可量化"通过 `evaluation/` 目录实现，完整方法论见 [evaluation/README.md](./evaluation/README.md)：

```text
标注集(qid + relevant_chunk_ids) --线上旁路采集(EVAL_DUMP_PATH)--> 检索 dump(jsonl)
        --run_retrieval_eval.py--> hit@5 / MRR / nDCG 报告
        --run_ablation.py--------> 消融对比表（full vs no_rrf / no_rerank / no_hyde / no_sparse）
```

- **线上零开销采集**：设置 `EVAL_DUMP_PATH` 后服务端自动把每次检索 TopK 与端到端耗时追加写入 jsonl；不设置则完全无感，采集失败也不影响主链路；
- **消融实验**：通过 `EVAL_ABLATION_MODE` 旁路开关裁剪链路（跳过精排 / 退化为单路 / 关闭 HyDE / 稀疏权重置 0），用同一份标注集对比各组件贡献，判断优化优先级；
- **badcase 闭环**：dump 保留完整排序，hit 未命中 → 查召回层，命中但靠后 → 查融合精排层，可溯源率低 → 查生成层 Prompt；
- 仓库内 `evaluation/results/*.sample.md` 为样例数据真实运行产出，仅演示格式；业务指标请用自己的标注集复现。

## A/B 实验框架

`app/experiments/` 提供线上对照实验的最小闭环，用于策略上线前的量化验证：

```python
from app.experiments.traffic_split import assign_variant

# 确定性哈希分流：同一会话永远命中同一分组，不同实验分组互不相关
variant = assign_variant(session_id, "rerank_latency_tradeoff", ratios=(0.9, 0.1))
```

- **确定性分流**：sha256(session_id + 实验名) 映射，无需存储分流关系，重启/扩容后依然成立，支持 90/10 灰度起步；
- **指标记录**：首字延迟 / 端到端耗时 / 自动命中判定 / 用户反馈写入 MongoDB（`ab_metrics` 集合），写入失败仅告警降级；
- **分组分析**：`python -m app.experiments.analyze --input ab_out.jsonl` 输出 p50/p95 延迟、命中率、反馈计数对比表；
- 与消融实验共用同一套链路旁路开关（`EVAL_ABLATION_MODE`），"离线消融定方向、线上 A/B 验收益"形成完整闭环。

## 依赖服务

运行前需准备以下服务（推荐直接使用上面的 `docker compose up -d`，无需手动准备）：

- Milvus 2.x（向量库）
- Neo4j 5.x（图数据库，社区版即可）
- MongoDB 6.x（文档库，同时存储 A/B 实验指标）
- MinIO（对象存储）

## License

MIT
