import os
import sys
from typing import List, Dict, Any
# 导入Milvus相关依赖
from pymilvus import DataType
# 导入自定义模块
from app.import_process.agent.state import ImportGraphState
from app.clients.milvus_utils import get_milvus_client
from app.utils.task_utils import add_running_task
from app.core.logger import logger
from app.conf.milvus_config import milvus_config
from app.utils.escape_milvus_string_utils import escape_milvus_string

# 从配置文件读取切片集合名称，与配置解耦，便于环境切换
CHUNKS_COLLECTION_NAME = milvus_config.chunks_collection

# ==========================================
# Milvus切片数据入库核心节点
# 核心能力：将上游向量化后的文本切片批量存入Milvus，实现幂等性写入
# 核心设计：
#   1. 幂等性：插入前删除同item_name旧数据，避免重复存储
#   2. 自动建表：集合不存在时自动创建Schema和向量索引，无需手动初始化
#   3. 数据校验：前置校验切片有效性、向量字段完整性，避免脏数据入库
#   4. 主键回填：将Milvus自增的chunk_id回填到切片，供下游业务使用
# 依赖上游：BGE-M3向量化节点（提供dense_vector/sparse_vector字段）
# ==========================================
def node_import_milvus(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph核心节点：Milvus切片数据入库主流程
    执行流程（串行执行，一步一校验，保证数据一致性）：
        1. 输入校验：验证切片有效性、向量字段完整性，提取向量维度
        2. 环境准备：连接Milvus，集合不存在则自动创建Schema+索引
        3. 幂等清理：删除同item_name旧数据，避免重复存储
        4. 批量插入：预处理数据后批量入库，回填Milvus自增chunk_id
        5. 状态更新：将回填了chunk_id的切片更新回全局状态，供下游使用
    参数：
        state: Dict[str, Any] - 流程全局状态对象，包含chunks、task_id等数据
    返回：
        Dict[str, Any] - 更新后的状态对象，chunks字段回填chunk_id
    异常处理：
        任一步骤失败抛出ValueError，终止节点执行，保证数据不脏写
    """
    # 获取当前节点名称，用于任务监控和日志标识
    current_node = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行LangGraph节点：{current_node}（Milvus切片数据入库）")
    # 标记任务运行状态，用于前端进度展示/任务监控
    add_running_task(state["task_id"], current_node)
    logger.info("--- Milvus切片数据入库流程启动 ---")

    try:
        # 步骤1：输入数据有效性校验
        chunks_json_data, vector_dimension = step_1_check_input(state)
        # 步骤2：Milvus客户端连接+集合准备（自动建表）
        client = step_2_prepare_collection(vector_dimension)
        # 步骤3：幂等性处理 - 清理同item_name旧数据
        step_3_clean_old_data(client, chunks_json_data)
        # 步骤4：批量插入数据+主键chunk_id回填
        updated_chunks = step_4_insert_data(client, chunks_json_data)
        # 步骤5：更新全局状态，将回填后的切片回传下游
        state["chunks"] = updated_chunks

        logger.info("--- Milvus切片数据入库流程完成 ---")
    except Exception as e:
        logger.error(f"Milvus切片数据入库节点执行失败：{str(e)}", exc_info=True)
        raise ValueError(f"Milvus 导入过程中发生错误: {e}")

    return state

def step_1_check_input(state: Dict[str, Any]) -> tuple[List[Dict[str, Any]], int]:
    """
    步骤1：输入数据有效性校验（入库前置必检）
    核心校验项：
        1. chunks非空且为列表类型
        2. 切片包含dense_vector核心字段（上游向量化节点必输）
        3. 提取向量维度，为集合创建/索引构建提供依据
    参数：
        state: Dict[str, Any] - 流程状态对象，包含上游传入的chunks数据
    返回：
        tuple - (校验通过的切片列表, 稠密向量维度)
    异常：
        任一校验项不通过，抛出ValueError终止入库流程，避免脏数据处理
    """
    # 提取待入库的切片数据
    chunks_json_data = state.get("chunks")
    # 校验1：chunks非空
    if not chunks_json_data:
        logger.error("Milvus入库校验失败：state中chunks字段为空")
        raise ValueError("错误: chunks为空，无法执行Milvus入库")
    # 校验2：chunks为非空列表
    if not isinstance(chunks_json_data, list) or len(chunks_json_data) == 0:
        logger.error("Milvus入库校验失败：chunks非列表类型或为空列表")
        raise ValueError("错误: chunks数据格式不正确，必须为非空列表")
    # 校验3：切片包含dense_vector字段（向量化节点核心产出）
    first_chunk = chunks_json_data[0]
    if 'dense_vector' not in first_chunk:
        logger.error("Milvus入库校验失败：切片缺失dense_vector字段，上游向量化节点可能执行失败")
        raise ValueError("错误: 数据中缺失dense_vector字段，请检查上游向量化节点执行状态")

    # 提取向量维度和商品名称，用于后续集合创建/日志展示
    vector_dimension = len(first_chunk['dense_vector'])
    item_name = first_chunk.get('item_name', '未知商品名')
    logger.info(
        f"Milvus入库校验通过，待入库切片数：{len(chunks_json_data)} | 向量维度：{vector_dimension} | 商品名称：{item_name}")

    return chunks_json_data, vector_dimension


def create_collection(client, collection_name: str, vector_dimension: int):
    """
    辅助函数：Milvus集合+索引自动创建
    核心逻辑：
        1. 定义集合Schema：包含业务字段+双向量字段，自增主键chunk_id
        2. 构建向量索引：稠密向量用AUTOINDEX（Milvus自动选最优索引），稀疏向量用专用索引
    参数：
        client - MilvusClient实例（已连接）
        collection_name: str - 要创建的集合名称
        vector_dimension: int - 稠密向量维度（与向量化模型保持一致）
    """
    # 1. 创建Schema：自增主键+支持动态字段，适配灵活的业务扩展
    schema = client.create_schema(auto_id=True, enable_dynamic_fields=True)

    # 2. 新增字段：业务字段+主键+双向量字段，字段类型/长度适配业务场景
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)  # 切片内容
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535, nullable=True)  # 切片标题
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535, nullable=True)  # 父标题
    schema.add_field(field_name="part", datatype=DataType.INT8, nullable=True)  # 分片编号
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535, nullable=True)  # 源文件标题
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535, nullable=True)  # 商品名称（幂等性依据）
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)  # 稀疏向量
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dimension)  # 稠密向量
    # 对于 BGE-M3 模型 ：
    # 它的输出维度是固定的 1024 。
    # 所以你的代码里必须是：
    # ```
    # vector_dimension=必须是1024，不能改！
    # schema.add_field(...,dim=vector_dimension)
    # ``` (如果你用的是 BGE-base ，那就是 768； BGE-small 是 384。这完全由模型架构决定。)
    # 3. 构建索引参数：为向量字段创建索引，提升检索性能
    index_params = client.prepare_index_params()
    # 优化版稠密向量索引：HNSW + COSINE (恢复最佳性能配置)
    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_vector_index",
        # HNSW (Hierarchical Navigable Small World) 是目前性能最好、最常用的基于图的索引，检索速度极快，精度极高。
        index_type="HNSW",
        # 使用 COSINE 作为稠密向量相似度计算方式
        metric_type="COSINE",
        # M: 图中每个节点的最大连接数(常用16-64)
        # efConstruction: 构建索引时的搜索范围(越大建索引越慢，但精度越高，常用100-200)
        params={"M": 16, "efConstruction": 200}
    )

    # 稀疏向量索引：专用SPARSE_INVERTED_INDEX+IP，关闭量化保证精度
    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_vector_index",
        # 稀疏倒排索引 专门为稀疏向量（比如文本的 TF-IDF 向量、关键词权重向量，特点是大部分元素为 0，只有少数维度有值）设计的倒排索引，是稀疏向量检索的标配索引类型。
        index_type="SPARSE_INVERTED_INDEX",
        # IP（内积，Inner Product）如果向量是 “文本语义向量 + 关键词权重”，长度代表文本与主题的关联强度，此时用 IP 能同时体现 “语义匹配度” 和 “关联强度”。
        metric_type="IP",
        # DAAT_MAXSCORE 是稀疏检索的高效算法，quantization="none" 保证稀疏向量权重无损失；normalize=是否归一化。
        params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"}
    )

    # 4. 创建集合：Schema+索引参数结合，一次性完成初始化
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    logger.info(f"Milvus集合创建成功：{collection_name}，向量维度：{vector_dimension}")


def step_2_prepare_collection(vector_dimension: int):
    """
    步骤2：Milvus客户端连接+集合准备（含Schema兼容性自动修复）
    核心逻辑：
        1. 获取Milvus单例客户端，验证连接有效性
        2. 集合不存在则自动创建（Schema+索引）
        3. 集合已存在则检查Schema兼容性：检测 parent_title/part 是否设置了 nullable
           - 兼容：直接复用
           - 不兼容：自动 drop + 重建（旧集合缺少 nullable，插入时会抛 DataNotMatchException）
    参数：
        vector_dimension: int - 稠密向量维度（步骤1提取）
    返回：
        MilvusClient - 已连接、集合准备完成的客户端实例
    异常：
        客户端获取失败/集合名称未配置，抛出ValueError终止流程
    """
    logger.info(f"开始准备Milvus环境，目标集合：{CHUNKS_COLLECTION_NAME}")
    # 1. 获取Milvus单例客户端，验证连接
    client = get_milvus_client()
    if client is None:
        logger.error("Milvus客户端获取失败：get_milvus_client()返回空，连接可能异常")
        raise ValueError("Milvus 连接失败：get_milvus_client() 返回空")
    # 2. 验证集合名称配置
    if not CHUNKS_COLLECTION_NAME:
        logger.error("Milvus集合名称未配置：CHUNKS_COLLECTION_NAME为空")
        raise ValueError("未配置CHUNKS_COLLECTION集合名称")

    # 3. 集合不存在则自动创建
    if not client.has_collection(collection_name=CHUNKS_COLLECTION_NAME):
        logger.info(f"Milvus集合{CHUNKS_COLLECTION_NAME}不存在，开始自动创建Schema和索引")
        create_collection(client, CHUNKS_COLLECTION_NAME, vector_dimension)
    else:
        logger.info(f"Milvus集合{CHUNKS_COLLECTION_NAME}已存在，开始Schema兼容性检查")
        # 检查旧集合Schema兼容性：若 parent_title / part 未设 nullable 则重建
        # 背景：旧版 Schema 未设置 nullable=True，插入缺失字段时会抛 DataNotMatchException
        if not _check_collection_schema_compatible(client, CHUNKS_COLLECTION_NAME):
            logger.warning(
                f"Milvus集合{CHUNKS_COLLECTION_NAME} Schema不兼容（parent_title/part缺少nullable属性），"
                f"自动删除旧集合并重建"
            )
            client.drop_collection(collection_name=CHUNKS_COLLECTION_NAME)
            create_collection(client, CHUNKS_COLLECTION_NAME, vector_dimension)
            logger.info(f"Milvus集合{CHUNKS_COLLECTION_NAME} Schema重建完成")
        else:
            logger.info(f"Milvus集合{CHUNKS_COLLECTION_NAME} Schema兼容，直接复用")

    return client


def _check_collection_schema_compatible(client, collection_name: str) -> bool:
    """
    辅助函数：检查集合 Schema 是否与当前代码定义兼容
    核心检查项：parent_title 和 part 字段是否设置了 nullable=True
    背景：旧版 Schema 这两个字段未设 nullable，插入缺失字段时会抛 DataNotMatchException
    参数：
        client - MilvusClient 实例
        collection_name: str - 要检查的集合名称
    返回：
        bool - True 表示兼容（可安全复用），False 表示需要重建
    """
    try:
        collection_info = client.describe_collection(collection_name=collection_name)
        fields = collection_info.get('fields', [])

        # 检查 parent_title 和 part 两个关键字段的 nullable 属性
        nullable_checks = {'parent_title': False, 'part': False}
        for field in fields:
            field_name = field.get('name', '')
            if field_name in nullable_checks:
                # describe_collection 返回的 nullable 可能是 bool 或 None（旧版本 API）
                is_nullable = field.get('nullable', False) or False
                nullable_checks[field_name] = is_nullable

        # 任一关键字段未设 nullable 则为不兼容
        for fname, is_nullable in nullable_checks.items():
            if not is_nullable:
                logger.warning(f"Schema兼容性检查：字段 {fname} 未设置 nullable=True，当前Schema不兼容")
                return False

        logger.info(f"Schema兼容性检查通过：parent_title 和 part 均设置了 nullable=True")
        return True
    except Exception as e:
        logger.warning(f"Schema兼容性检查失败：{e}，保守处理视为不兼容")
        return False


def step_3_clean_old_data(client, chunks_json_data: List[Dict[str, Any]]):
    """
    步骤3：幂等性处理 - 基于item_name清理旧数据
    核心设计：
        插入新数据前删除同item_name的所有旧切片，确保多次执行仅保留最新数据
        支持多item_name批量清理，自动去重避免重复操作
    参数：
        client - MilvusClient实例
        chunks_json_data: List[Dict[str, Any]] - 待入库的切片列表
    """
    # 提取并去重item_name，避免重复清理同一商品数据
    # - 顺序 ：先循环 ( for ) -> 再判断 ( if ) -> 最后产出 ( name )。
    # - 海象操作符 ( := ) 的作用 ：它在第 ② 步判断的时候，顺手把处理好的字符串塞进了 name 变量里。如果 name 是空字符串 ""
    # （在 Python 里等同于 False）， if 条件不成立，第 ③ 步就不会执行，这个空值就被扔掉了。
    item_names = sorted(
    {   name  # ③ 最后一步：如果没被 if 拦住，把 name 丢进篮子里
        for x in chunks_json_data or []  # ① 第一步：开始循环，拿到 x
        if (name := str(x.get("item_name", "")).strip())  # ② 第二步：提取 -> 去空格 -> 赋值给 name -> 判断 name 是否为空
    })

    # 无有效item_name则跳过清理
    if not item_names:
        logger.warning("Milvus幂等性清理跳过：切片中无有效item_name")
        return
    # 多item_name提示日志
    if len(item_names) > 1:
        logger.warning(f"Milvus幂等性清理：本次检测到多个item_name，将逐个清理：{item_names}")

    # 遍历item_name，逐个清理旧数据
    for i_name in item_names:
        _clear_chunks_by_item_name(client, CHUNKS_COLLECTION_NAME, i_name)


def _clear_chunks_by_item_name(client, collection_name: str, item_name: str):
    """
    内部核心函数：根据item_name删除Milvus中的旧切片数据
    参数：
        client - MilvusClient实例
        collection_name: str - 集合名称
        item_name: str - 要清理的商品名称
    异常：
        清理失败抛出ValueError，终止整个入库流程（保证幂等性）
    """
    # 预处理：去除空格，空值直接返回
    i_name = (item_name or "").strip()
    if not i_name:
        logger.warning("Milvus单商品清理跳过：item_name为空")
        return
    if not collection_name:
        logger.warning("Milvus单商品清理跳过：集合名称未配置")
        return

    try:
        # 集合不存在则无需清理
        if not client.has_collection(collection_name=collection_name):
            logger.info(f"Milvus单商品清理跳过：集合{collection_name}不存在")
            return

        # 1. 商品名称安全转义，避免filter表达式报错
        safe_item_name = escape_milvus_string(i_name)
        filter_expr = f'item_name == "{safe_item_name}"'
        logger.info(f"Milvus幂等性清理：开始删除集合{collection_name}中item_name={i_name}的旧数据")

        # 2. 执行删除操作
        client.delete(collection_name=collection_name, filter=filter_expr)

        # 3. 强制flush，确保删除操作立即生效（避免Milvus异步延迟）
        if hasattr(client, "flush"):
            try:
                client.flush(collection_name=collection_name)
            except Exception as e:
                logger.warning(f"Milvus幂等性清理：flush操作失败，不影响主流程 | 错误：{str(e)}")

        logger.info(f"Milvus幂等性清理完成：成功删除item_name={i_name}的旧数据")
    except Exception as e:
        logger.error(f"Milvus幂等性清理失败：item_name={i_name} | 错误：{str(e)}", exc_info=True)
        raise ValueError(f"幂等清理失败（item_name={i_name}）: {e}")

def step_4_insert_data(client, chunks_json_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    步骤4：批量插入切片数据到Milvus+主键回填
    核心逻辑：
        1. 数据清洗：补全缺失字段、移除手动chunk_id，确保符合Schema要求
        2. 批量插入数据：提升入库效率，减少Milvus连接次数
        3. 回填chunk_id：将Milvus生成的自增主键回填到切片，供下游业务使用
    参数：
        client - MilvusClient实例
        chunks_json_data: List[Dict[str, Any]] - 待入库的切片列表
    返回：
        List[Dict[str, Any]] - 回填了chunk_id的切片列表
    """
    # 0. 数据清洗：确保每条记录都包含Schema要求的字段，避免插入时DataNotMatchException
    data_to_insert = []
    # Schema字段默认值映射（与create_collection中的字段定义保持一致）
    field_defaults = {
        "content": "",
        "title": "",
        "parent_title": "",
        "part": 0,
        "file_title": "",
        "item_name": "",
    }
    for item in chunks_json_data:
        if not isinstance(item, dict):
            logger.warning(f"Milvus入库数据清洗：跳过非字典类型切片，类型={type(item)}")
            continue
        item_copy = item.copy()
        # 移除手动chunk_id，避免与Milvus自增主键冲突
        item_copy.pop("chunk_id", None)
        # 补全缺失字段：字段不存在或值为None时，填充默认值
        for field, default_val in field_defaults.items():
            if field not in item_copy or item_copy[field] is None:
                item_copy[field] = default_val
        data_to_insert.append(item_copy)

    if not data_to_insert:
        logger.error("Milvus数据插入失败：清洗后无有效数据可插入")
        raise ValueError("Milvus插入数据为空：所有切片均非字典类型或已被过滤")

    logger.info(f"Milvus数据插入：准备{len(data_to_insert)}条切片数据，开始批量插入")
    # 1. 执行批量插入
    insert_result = client.insert(collection_name=CHUNKS_COLLECTION_NAME, data=data_to_insert)
    insert_count = insert_result.get('insert_count', 0)
    logger.info(f"Milvus数据插入完成：成功插入{insert_count}条数据，插入结果：{insert_result}")

    # 2. 主键回填：将Milvus生成的chunk_id回填到原始切片
    inserted_ids = insert_result.get('ids', [])
    if inserted_ids and len(inserted_ids) == len(chunks_json_data):
        logger.info(f"Milvus主键回填：开始将{len(inserted_ids)}个自增chunk_id回填到切片")
        for idx, item in enumerate(chunks_json_data):
            item['chunk_id'] = str(inserted_ids[idx])
        logger.info("Milvus主键回填完成：所有切片已绑定chunk_id")
    else:
        logger.warning(f"Milvus主键回填失败：生成ID数量({len(inserted_ids)})与切片数量({len(chunks_json_data)})不一致")

    return chunks_json_data


if __name__ == '__main__':
    # --- 单元测试 ---
    # 目的：验证 Milvus 导入节点的完整流程，包括连接、创建集合、清理旧数据和插入新数据。
    import sys
    import os
    from dotenv import load_dotenv

    # 加载环境变量 (自动寻找项目根目录的 .env)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造测试数据
    dim = 1024
    test_state = {
        "task_id": "test_milvus_task",
        "chunks": [
            {
                "content": "Milvus 测试文本 1",
                "title": "测试标题",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title":"test.pdf",
                "part":1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
        ]
    }

    print("正在执行 Milvus 导入节点测试...")
    try:
        # 检查必要的环境变量
        if not os.getenv("MILVUS_URL"):
            print("❌ 未设置 MILVUS_URL，无法连接 Milvus")
        elif not os.getenv("CHUNKS_COLLECTION"):
            print("❌ 未设置 CHUNKS_COLLECTION")
        else:
            # 执行节点函数
            result_state = node_import_milvus(test_state)

            # 验证结果
            chunks = result_state.get("chunks", [])
            if chunks and chunks[0].get("chunk_id"):
                print(f"✅ Milvus 导入测试通过，生成 ID: {chunks[0]['chunk_id']}")
            else:
                print("❌ 测试失败：未能获取 chunk_id")

    except Exception as e:
        print(f"❌ 测试失败: {e}")