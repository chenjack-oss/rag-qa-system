# Graph RAG 补齐路线图（rag-qa-system）

> 现状：检索工作流已预留 `node_query_kg` 节点入口和 `kg_chunks` 状态字段，Neo4j 连接配置已就绪，
> 但图谱构建（导入侧）、图谱查询（检索侧）、结果融合（排序侧）三层均为空实现。
> 本路线图按依赖顺序分四个阶段，每阶段独立可交付、可测试、可单独提交。

---

## 为什么 rag-qa-system 需要 Graph RAG（面试话术）

向量检索的三个固有缺陷，恰好是知识图谱的强项：

| 场景 | 向量检索的问题 | KG 的解法 |
|------|--------------|----------|
| **多跳问题**：“烫金机温度传感器的供应商是谁” | 分块嵌入后，传感器和供应商大概率不在同一个 chunk，语义相似度召回不到 | 实体间路径直达：温度传感器 → 属于 → 烫金机 → 供应商 → X |
| **精确参数**：“HAK 180 的最大电压” | 220V 和 380V 在向量空间几乎重合，容易召回错型号的参数 | 实体节点带属性，按 item_name 精确过滤 |
| **实体消歧**：“P60” 匹配到多个型号 | 向量无法区分同名实体 | 图中实体挂在商品节点下，按 item_name 子图隔离 |

工程定位：**KG 不是替换向量检索，是第五路召回的补充信号源**——精确关系事实走 KG，
模糊语义相似走向量，RRF 融合后由 Reranker 统一裁决。

---

## 阶段一：导入侧 —— 图谱构建链路（地基，最高优先级）

图里没数据，检索侧无从谈起。本阶段让“导入一个文档”同时产出向量库 + 知识图谱。

### 交付物

**1. `prompts/entity_relation_extraction.prompt`（新增）**

LLM 三元组抽取模板，输出约束为 JSON 数组：

```
[{"head": "温度传感器", "head_type": "部件", "relation": "属于", "tail": "HAK 180 烫金机", "tail_type": "产品"}, ...]
```

要点：
- 限定业务实体类型（产品/部件/功能/参数/故障/耗材），防止 LLM 抽出垃圾实体
- 明确“只抽取文本中明确陈述的关系，禁止推测”
- 限制单 chunk 最多抽 10 条，防止关系爆炸

**2. `app/clients/neo4j_utils.py`（扩展 13 行 → 约 150 行）**

```python
def get_neo4j_driver()          # 已有，补充连接失败降级
def create_constraints()        # CREATE CONSTRAINT IF NOT EXISTS：entity.name 唯一索引
def batch_merge_triples(triples, source_chunk_id, item_name)
    # UNWIND + MERGE 幂等写入：
    # MERGE (h:Entity {name: $head})
    #   ON CREATE SET h.type = $head_type
    # MERGE (t:Entity {name: $tail})
    #   ON CREATE SET t.type = $tail_type
    # MERGE (h)-[r:REL {type: $relation}]->(t)
    #   ON CREATE SET r.source_chunk_ids = [$chunk_id]
    #   ON MATCH  SET r.source_chunk_ids = coalesce(r.source_chunk_ids,[]) + $chunk_id
    # 关键：实体挂 item_name 属性做商品子图隔离（解决消歧）
def query_entity_neighborhood(entity_name, item_name, max_hops=2, limit=20)  # 阶段二用
```

**3. `app/import_process/agent/nodes/node_kg_build.py`（新增节点）**

仿照 `node_item_name_recognition.py` 的六步结构：

```
step_1_get_inputs          # 取 chunks + item_name（复用上游产物）
step_2_build_context       # 每次取 N 个 chunk 拼上下文（控制 token）
step_3_call_llm            # 逐批抽取三元组，JSON 解析容错（修复 markdown 代码块包裹）
step_4_deduplicate         # 三元组去重（head,relation,tail）元组级
step_5_write_neo4j         # batch_merge_triples 幂等写入
step_6_update_state        # state["kg_triples_count"] 统计回填
```

**4. 接入 `import_process/agent/main_graph.py`**

```
node_item_name_recognition → node_kg_build → node_bge_embedding → node_import_milvus
```

（KG 构建依赖 item_name，放在识别节点之后；与 BGE 向量化无依赖，未来可并行）

**5. `ImportGraphState` 补字段**：`kg_triples_count: int`

### 验收标准
- 导入一份说明书后，Neo4j Browser 里能看到 Entity 节点和关系
- 重复导入同一文档，节点/关系数量不变（幂等）
- Neo4j 不可用时流程不中断，只记 warning（与 Milvus 同样的降级风格）

---

## 阶段二：检索侧 —— node_query_kg 实装（消灭 sleep 占位符）

### 交付物

**1. `node_query_kg.py` 重写（16 行 → 约 200 行）**

```
step_1_extract_entities    # LLM 从 query 抽实体（复用阶段一的抽取 prompt 简化版），
                           # 兜底：直接拿 item_name + query 中的名词短语
step_2_link_entities       # 实体链接：Neo4j 里模糊匹配（CONTAINS/正则），返回图内标准实体名
step_3_query_subgraph      # query_entity_neighborhood：1~2 跳邻居
step_4_paths_to_chunks     # 路径转文本：“HAK 180 的温度传感器 属于 HAK 180 烫金机”，
                           # 组装成与向量召回同构的 {chunk_id, content, item_name}
step_5_update_state         # state["kg_chunks"] = [...]
```

**2. Prompt：`prompts/query_entity_extraction.prompt`（新增）**

### 验收标准
- 问“X 的部件有哪些”，kg_chunks 返回正确的路径文本
- 图里查不到实体时返回空列表，不报错

---

## 阶段三：融合侧 —— KG 结果进入排序管线

现在 `node_rrf` 只融合 embedding + HyDE 两路；`node_rerank.step_1_merge_docs` 只合并 local + web。

### 交付物（二选一，推荐 A）

**方案 A（推荐）：KG 作为第三路进 RRF**
```python
source_weights = [
    (embedding_chunks, 1.0),
    (hyde_embedding_chunks, 1.0),
    (kg_chunks, 0.8),   # KG 精确但覆盖窄，权重略降
]
```
理由：KG chunk 与向量 chunk 同构（都有 chunk_id），天然可进 RRF；RRF 按排名融合，
KG 结果少不是问题。

**方案 B：走 rerank merge，source="kg"**
改动小，但绕过了 RRF，KG 和向量结果可能出现重复内容。

**配套改动**：`step_1_merge_docs` 加 kg 分支 + `step_3_topk` 断崖逻辑天然兼容多源。

### 验收标准
- 端到端：问多跳问题，最终 reranked_docs 里出现 KG 路径文本
- RRF 单测更新：三路融合权重用例

---

## 阶段四：文档 + 单元测试（面试弹药）

**1. `docs/graph_rag.md`（新增设计文档）**
- 架构图：导入链路（chunk → LLM 抽取 → 三元组 → Neo4j）
- 设计决策：为什么用 MERGE 幂等、为什么实体挂 item_name、为什么 KG 走 RRF 不走 merge
- 踩坑记录：LLM 抽取的常见 badcase（推测关系、类型漂移）及 prompt 迭代过程

**2. 单元测试（纯函数，不依赖 Neo4j 实例）**
- `test_triple_parser.py`：LLM 返回 JSON 的容错解析（markdown 包裹/截断/字段缺失）
- `test_paths_to_chunks.py`：路径转文本
- `test_rrf.py` 扩展：三路融合
- neo4j_utils 的 Cypher 构造可用 mock driver 验证

**3. README 更新**：检索链路图四路 → 五路，加 Graph RAG 章节

---

## 提交切分建议（每个独立 PR/commit）

| # | commit 主题 | 阶段 |
|---|------------|------|
| 1 | `feat(kg): 实体关系抽取 prompt + 三元组解析与容错` | 一 |
| 2 | `feat(kg): Neo4j 幂等写入（UNWIND+MERGE）与约束管理` | 一 |
| 3 | `feat(kg): 导入工作流新增图谱构建节点` | 一 |
| 4 | `feat(kg): 检索侧 node_query_kg 实装（实体链接+子图查询）` | 二 |
| 5 | `feat(kg): KG 三路 RRF 融合 + rerank 多源合并` | 三 |
| 6 | `docs(kg): Graph RAG 设计文档 + 测试补全` | 四 |

每个 commit 都是真实工程改进，单独可回滚，绿格子是副产品不是目的。

---

## 本地开发环境（跑通前提）

```bash
# Neo4j 可用 Docker 一键起：
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/yourpassword neo4j:5-community
# .env 里填 NEO4J_URI=bolt://localhost:7687 即可
```
