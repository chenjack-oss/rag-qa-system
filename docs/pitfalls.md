# 工程踩坑总结（pitfalls）

> 本文档记录项目开发与 CI 建设过程中真实踩过的坑与最终解法，持续更新。
> 相关背景可结合 [architecture.md](./architecture.md) 阅读。

## 1. CI 干净检出：`.env` 缺失导致导入即崩

**现象**：GitHub Actions 干净 clone 后，部分测试在 import 阶段直接抛 `FileNotFoundError`，本地却全部通过。

**根因**：`app/utils/path_utils.get_project_root()` 的定位策略是「向上逐级查找 `.env` 文件」来锚定项目根目录。本地开发环境根目录存在 `.env`（含各种凭据），但 `.env` 不入库，CI 检出后该文件不存在——向上查找到文件系统根也找不到，导入即崩。这类问题的隐蔽性在于：**崩溃点往往不在业务代码本身，而在被间接 import 的工具模块**，报错栈离真正的根因很远。

**解法**：`get_project_root()` 的设计里，环境变量 `PROJECT_ROOT` 是第一优先级分支。于是不动任何生产代码，在 `tests/conftest.py` 注入：

```python
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
```

要点：

- 用 `setdefault` 而非直接赋值，不覆盖用户已有配置；
- conftest 会被 pytest 在收集任何测试前自动导入，保证时序；
- 生产环境不受影响（仍然走 env var → 向上查找的原有顺序）。

**教训**：配置查找逻辑如果依赖「仓库外隐式文件」，必须给 CI/容器环境留一条显式注入通道（环境变量优先级最高），否则干净环境必然踩坑。

## 2. 重依赖模块级 import 导致节点不可测

**现象**：想给检索链路的 rerank / 查询项确认节点写单测，发现测试文件一 import 节点模块就挂。

**根因**：这些节点模块（`node_rerank.py`、`node_item_name_confirm.py` 等）在**模块顶部** import 了 `FlagEmbedding`（会级联加载 torch，数百 MB）、`pymilvus`、`pymongo`、`langchain_core`。而 CI（`ci.yml`）刻意只安装轻量依赖（`pytest numpy loguru fastapi python-dotenv`），不装任何深度学习/数据库 SDK——CI 环境里这些模块**根本无法导入**，连带所有引用它们的测试一起阵亡。

**解法**：把重依赖从模块级移到**真正使用它的函数体内**（延迟导入）。例如 `node_rerank.py`：

```python
def step_2_rerank_docs(state, doc_items):
    # 延迟导入：FlagEmbedding 会级联加载 torch 等重量级依赖
    from app.lm.reranker_utils import get_reranker_model
    ...
```

**取舍分析**（为什么敢这么改）：

- 延迟导入的代价是「错误从 import 时提前暴露变成首次调用时才暴露」，对本项目可接受：这些重依赖只在检索执行路径上使用，而执行路径本身就需要 Milvus/LLM 等外部服务在线，import 失败和连接失败属于同一类「运行时环境问题」；
- 换来的收益是节点内**纯逻辑部分**（文档合并、TopK 截断、断崖阈值判定、提示词构造等）可以在 CI 中零依赖测试——这部分恰恰是最值得测的分支逻辑；
- 修改后这些节点的单测从 0 到 19+21+17+ 大量用例，全仓库测试 97 → 168 passed。

**教训**：重依赖应该收敛到「客户端封装层」（本项目是 `app/clients/`、`app/lm/`），编排/节点层保持对重依赖的延迟引用，测试性会好一个数量级。

## 3. CI 等价自检：`sitecustomize` 屏蔽重依赖

**背景**：推送前想在本机验证「CI 会长什么样」，但本地装了全部重依赖，直接跑 pytest 通过了也无法证明 CI 能通过。

**做法**：在临时目录放一个 `sitecustomize.py`，用 `sys.meta_path` 把重依赖模块整体屏蔽：

```python
# 临时 sitecustomize.py（验证完删除，不入库）
import sys

class Blocker:
    blocked = {"pymilvus", "pymongo", "bson", "langchain", "torch", "FlagEmbedding"}
    def find_module(self, name, path=None):
        if name.split(".")[0] in self.blocked:
            return self
    def load_module(self, name):
        raise ImportError(f"blocked in CI-equivalence check: {name}")

sys.meta_path.insert(0, Blocker())
```

然后 `PYTHONPATH=<临时目录> pytest tests/ -q`。更严格的做法是再把仓库 `git clone` 到临时目录（天然无 `.env`），模拟干净检出跑一遍。

**效果**：本地即可复现「CI 无重依赖 + 无 .env」的双重环境，多次在推送前拦住了「本地绿、CI 必红」的提交。

## 4. Windows 下 venv 的脚本目录是 `Scripts` 不是 `bin`

在 Windows 上创建的 venv，pytest / ruff 等可执行文件位于 `.venv/Scripts/`（Linux/macOS 是 `.venv/bin/`）。文档或 CI 脚本里写死 `bin/` 会在 Windows 直接找不到命令。写跨平台说明或脚本时需区分处理。

## 5. lint 范围要显式覆盖 tests/

CI 早期 ruff 只检查 `app/`，测试代码没有任何 lint 约束，慢慢积累了 import 排序、未使用变量等问题。后来把 ci.yml 的 lint 范围从 `app/` 扩展为 `app/ tests/`，一次修复存量问题后，新增代码不再劣化。**测试代码同样值得 lint**——它是长期资产。

## 6. 评测口径必须先冻结，再看指标

做检索评测（hit@k / MRR / nDCG）时踩过的坑：评测集口径不固定（比如「可答问题」的边界反复调整），不同轮次的指标不可比，调参等于盲调。最终固定为：**冻结 fixed 评测集口径 → 每组配置跑 3 轮取均值 → 消融实验逐组件开关**，配合 RRF(k=60) 带权融合与三级置信过滤的调参记录，保证每处指标变化都能归因到具体改动。
