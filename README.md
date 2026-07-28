# RAG知识问答系统

基于 LangGraph 构建的企业级知识库系统，覆盖「文档入库 → 多路检索 → 重排 → 生成」完整链路，支持 PDF/Word 等非结构化文档的解析、向量化、图谱建模与多路召回问答。

## 项目特性

- **文档导入流水线**：MinerU 解析 PDF → 文档切片 → BGE-M3 向量化 → Milvus 入库 → 实体识别 → Neo4j 图谱建模
- **多路检索问答**：向量检索（Milvus）+ 图谱检索（Neo4j）联合召回，BGE-reranker-large 重排序，Qwen 大模型生成
- **工程化设计**：配置与代码分离（`app/conf/`）、客户端统一封装（`app/clients/`）、LangGraph 编排导入流程
- **可观测**：内置分级日志（控制台 + 文件，按天滚动）

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
│   ├── import_process/   # 文档导入流程
│   │   ├── agent/        # LangGraph 导入图（nodes + state）
│   │   └── api/          # 文件导入服务
│   ├── lm/               # 大模型工具（embedding/reranker/lm 调用）
│   ├── query_process/    # 查询问答流程
│   ├── tool/             # 模型下载等辅助工具
│   └── utils/            # 通用工具
├── prompts/              # 提示词模板
├── test/                 # 测试与示例
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

### 4. 启动服务

```bash
python -m app.import_process.api.file_import_service
```

## 依赖服务

运行前需自行准备以下服务：

- Milvus 2.x（向量库）
- Neo4j 5.x（图数据库，社区版即可）
- MongoDB 6.x（文档库）
- MinIO（对象存储）

## License

MIT
