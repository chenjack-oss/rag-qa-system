"""批量回放标注集：逐条调用在线查询服务，服务端自动 dump 检索结果

前置条件：
1. 查询服务已启动（python -m app.query_process.api.query_service 或 docker compose up）
2. 服务进程设置了 EVAL_DUMP_PATH=/path/to/dump.jsonl（采集开关）
3. 做消融时，服务进程需按配置设置 EVAL_ABLATION_MODE=no_rrf|no_rerank|no_hyde|no_sparse
   （一次进程只跑一个配置；full 为默认）

用法：
    python evaluation/dump_retrieval.py \
        --dataset evaluation/data/eval_dataset.sample.jsonl \
        --api http://127.0.0.1:8001/query \
        --concurrency 3
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib import error, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def replay_one(api_url: str, item: dict, timeout: int) -> dict:
    payload = json.dumps({
        "query": item["query"],
        "session_id": item.get("qid", ""),
        "is_stream": False,
    }).encode("utf-8")
    req = request.Request(api_url, data=payload, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return {
        "qid": item.get("qid", ""),
        "ok": True,
        "answer_len": len(body.get("answer", "") or ""),
        "elapsed_s": round(time.perf_counter() - start, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="标注集批量回放采集")
    parser.add_argument("--dataset", required=True, help="标注集 jsonl（须含 qid/query 字段）")
    parser.add_argument("--api", default="http://127.0.0.1:8001/query", help="查询服务 /query 地址")
    parser.add_argument("--concurrency", type=int, default=3, help="并发数（默认 3，避免打爆 Embedding 串行推理）")
    parser.add_argument("--timeout", type=int, default=120, help="单条超时秒数")
    parser.add_argument("--report", default="", help="回放汇总输出路径（json）")
    args = parser.parse_args()

    items = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                if item.get("query"):
                    items.append(item)

    print(f"回放样本数: {len(items)}, 并发: {args.concurrency}, API: {args.api}")
    results = []
    failed = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(replay_one, args.api, item, args.timeout): item for item in items}
        for i, fut in enumerate(as_completed(futures), 1):
            item = futures[fut]
            try:
                res = fut.result()
                results.append(res)
                print(f"[{i}/{len(items)}] {res['qid']} ok {res['elapsed_s']}s")
            except (error.URLError, error.HTTPError, TimeoutError, KeyError) as e:
                failed.append({"qid": item.get("qid", ""), "error": str(e)})
                print(f"[{i}/{len(items)}] {item.get('qid', '')} FAILED: {e}")

    summary = {
        "total": len(items),
        "success": len(results),
        "failed": len(failed),
        "avg_elapsed_s": round(sum(r["elapsed_s"] for r in results) / len(results), 2) if results else 0.0,
        "failures": failed,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, ensure_ascii=False))
    if args.report:
        Path(args.report).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"汇总已写入: {args.report}")


if __name__ == "__main__":
    main()
