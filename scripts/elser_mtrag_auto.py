#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
elser_mtrag_auto_sleep.py

- ES 8.10 + ELSER v1 (.elser_model_1) baseline runner for mtRAG
- bulk action 显式包含: {"index":{"_index": idx, "_id": doc_id, "pipeline": pipeline_id}}
- 默认 resume：用 create op，已存在文档 409 冲突会被忽略（可随时停/重跑）
"""

import argparse
import gzip
import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# -------------------------
# small helpers
# -------------------------
def open_maybe_gz(path: str, mode: str = "rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", errors="ignore")
    return open(path, mode, encoding="utf-8", errors="ignore")

def load_blacklist(path: Optional[str]) -> set:
    if not path:
        return set()
    bl = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s:
                bl.add(s)
    print(f"[BLACKLIST] loaded {len(bl)} ids from {path}")
    return bl

def json_dumps(x) -> str:
    return json.dumps(x, ensure_ascii=False)

# -------------------------
# minimal ES HTTP client (stdlib only)
# -------------------------
class ESClient:
    def __init__(self, base_url: str, timeout_s: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _url(self, path: str, params: Optional[dict] = None) -> str:
        if not path.startswith("/"):
            path = "/" + path
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params, doseq=True)
        return url

    def request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        body_obj: Optional[dict] = None,
        body_bytes: Optional[bytes] = None,
        headers: Optional[dict] = None,
        ok_codes: Tuple[int, ...] = (200, 201),
        retry: int = 0,
        retry_sleep_s: float = 2.0,
    ) -> dict:
        url = self._url(path, params=params)
        hdr = {"Accept": "application/json"}
        if headers:
            hdr.update(headers)

        data = None
        if body_bytes is not None:
            data = body_bytes
        elif body_obj is not None:
            payload = json.dumps(body_obj).encode("utf-8")
            data = payload
            hdr.setdefault("Content-Type", "application/json")

        last_err = None
        for attempt in range(retry + 1):
            try:
                req = urllib.request.Request(url, data=data, headers=hdr, method=method.upper())
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    code = resp.getcode()
                    raw = resp.read()
                if code not in ok_codes:
                    raise RuntimeError(f"HTTP {code} for {method} {url}: {raw[:300]}")
                if not raw:
                    return {}
                return json.loads(raw.decode("utf-8", errors="ignore"))
            except Exception as e:
                last_err = e
                if attempt < retry:
                    time.sleep(retry_sleep_s)
                    continue
                raise last_err

    def get(self, path: str, **kw) -> dict:
        return self.request("GET", path, **kw)

    def put(self, path: str, **kw) -> dict:
        return self.request("PUT", path, **kw)

    def post(self, path: str, **kw) -> dict:
        return self.request("POST", path, **kw)

    def delete(self, path: str, **kw) -> dict:
        return self.request("DELETE", path, **kw)

# -------------------------
# mtRAG loading
# -------------------------
def load_queries(query_path: str) -> Dict[str, str]:
    queries = {}
    with open(query_path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            qid = o.get("_id") or o.get("query_id") or o.get("task_id")
            queries[qid] = o["text"]
    return queries

def iter_corpus_docs(corpus_path: str, blacklist: set) -> Iterable[Tuple[str, str]]:
    """
    yields (doc_id, text)
    text = title + "\n\n" + text
    """
    with open_maybe_gz(corpus_path, "rt") as f:
        for line in f:
            o = json.loads(line)
            doc_id = str(o.get("document_id") or o.get("_id") or o.get("id") or "")
            if not doc_id:
                continue
            if doc_id in blacklist:
                continue
            text = (f'{o.get("title","")}\n\n{o.get("text","")}'.strip())
            yield doc_id, text

def count_corpus_docs(corpus_path: str, blacklist: set) -> int:
    n = 0
    for _doc_id, _text in iter_corpus_docs(corpus_path, blacklist):
        n += 1
    return n

# -------------------------
# ELSER v1 setup
# -------------------------
def ensure_trial(es: ESClient):
    try:
        r = es.post("/_license/start_trial", params={"acknowledge": "true"}, retry=1)
        print("[TRIAL] start_trial response:", r)
    except Exception as e:
        print("[TRIAL] skip (likely already active or not permitted):", str(e)[:200])

def ensure_elser_v1_model(es: ESClient, model_id: str):
    try:
        es.get(f"/_ml/trained_models/{urllib.parse.quote(model_id)}")
        print(f"[MODEL] exists: {model_id}")
    except Exception:
        print(f"[MODEL] creating (will download if needed): {model_id}")
        es.put(
            f"/_ml/trained_models/{urllib.parse.quote(model_id)}",
            body_obj={"input": {"field_names": ["text_field"]}},
            retry=2,
            retry_sleep_s=3.0,
        )
        print(f"[MODEL] created: {model_id}")

def ensure_deployment_started(es: ESClient, model_id: str, deployment_id: str):
    print(f"[DEPLOY] starting deployment: model={model_id} deployment_id={deployment_id}")
    try:
        es.post(
            f"/_ml/trained_models/{urllib.parse.quote(model_id)}/deployment/_start",
            params={
                "deployment_id": deployment_id,
                "wait_for": "started",
                "timeout": "30m",
            },
            retry=2,
            retry_sleep_s=5.0,
        )
        print("[DEPLOY] started (or already usable)")
    except Exception as e:
        print("[DEPLOY] start skipped/failed (may already be started):", str(e)[:250])

def update_deployment_allocations(es: ESClient, model_id: str, number_of_allocations: int):
    # Official endpoint: POST /_ml/trained_models/{model_id}/deployment/_update
    # (If ES refuses, it会返回 max_assigned_allocations=1 之类的信息)
    if number_of_allocations <= 0:
        return
    try:
        r = es.post(
            f"/_ml/trained_models/{urllib.parse.quote(model_id)}/deployment/_update",
            body_obj={"number_of_allocations": number_of_allocations},
            ok_codes=(200,),
        )
        tp = (r.get("assignment", {}) or {}).get("task_parameters", {}) or {}
        ma = (r.get("assignment", {}) or {}).get("max_assigned_allocations", None)
        print(f"[DEPLOY-UPDATE] requested_alloc={number_of_allocations} "
              f"-> current/target maybe={tp.get('number_of_allocations')} max_assigned={ma}")
    except Exception as e:
        print("[DEPLOY-UPDATE] failed (ignore):", str(e)[:200])

def ensure_pipeline(es: ESClient, pipeline_id: str, model_id: str):
    body = {
        "description": "mtRAG ELSER v1 indexing pipeline",
        "processors": [
            {
                "inference": {
                    "model_id": model_id,
                    "target_field": "ml",
                    "field_map": {"text": "text_field"},
                    "inference_config": {"text_expansion": {"results_field": "tokens"}},
                }
            }
        ],
    }
    es.put(f"/_ingest/pipeline/{urllib.parse.quote(pipeline_id)}", body_obj=body)
    print(f"[PIPELINE] ready: {pipeline_id}")

def ensure_index(es: ESClient, index_name: str, pipeline_id: str, recreate: bool):
    if recreate:
        try:
            es.delete(f"/{urllib.parse.quote(index_name)}", ok_codes=(200, 404))
            print(f"[INDEX] deleted: {index_name}")
        except Exception as e:
            print(f"[INDEX] delete error (ignore): {index_name} | {str(e)[:200]}")

    body = {
        "settings": {
            "index": {
                "default_pipeline": pipeline_id,
                "number_of_shards": 1,
                "number_of_replicas": 0,
                # bulk 提速（可选）：refresh_interval=-1，bulk 结束后我们会恢复
                "refresh_interval": "1s",
            }
        },
        "mappings": {
            "properties": {
                "document_id": {"type": "keyword"},
                "text": {"type": "text"},
                "ml": {"properties": {"tokens": {"type": "rank_features"}}},
            }
        },
    }
    try:
        es.put(f"/{urllib.parse.quote(index_name)}", body_obj=body, ok_codes=(200, 201))
        print(f"[INDEX] created: {index_name}")
    except Exception as e:
        msg = str(e)
        if "resource_already_exists_exception" in msg or "already exists" in msg:
            print(f"[INDEX] exists: {index_name}")
        else:
            raise

def set_refresh_interval(es: ESClient, index_name: str, refresh_interval: str):
    es.put(f"/{urllib.parse.quote(index_name)}/_settings",
           body_obj={"index": {"refresh_interval": refresh_interval}},
           ok_codes=(200,))
    print(f"[INDEX] {index_name} refresh_interval={refresh_interval}")

def get_index_count(es: ESClient, index_name: str) -> int:
    r = es.get(f"/{urllib.parse.quote(index_name)}/_count", ok_codes=(200,))
    return int(r.get("count", 0))

def bulk_index(
    es: ESClient,
    index_name: str,
    pipeline_id: str,
    corpus_path: str,
    blacklist: set,
    batch_docs: int = 500,
    resume: bool = True,
    fast_bulk: bool = True,
):
    """
    - resume=True: 使用 create op，已存在 doc -> 409 conflict，忽略
    - 显式在 action meta 写 _index/_id/pipeline（你要求的可见格式）
    """
    print(f"[BULK] indexing -> {index_name} (resume={resume}, fast_bulk={fast_bulk})")

    total_expected = count_corpus_docs(corpus_path, blacklist)
    already = get_index_count(es, index_name)
    print(f"[BULK] expected_after_blacklist={total_expected} already_in_index={already} remaining~={max(total_expected - already, 0)}")

    if fast_bulk:
        set_refresh_interval(es, index_name, "-1")

    n_sent = 0
    n_ok = 0
    n_conflict = 0
    t0 = time.time()
    ndjson_lines: List[str] = []

    def flush():
        nonlocal ndjson_lines, n_ok, n_conflict
        if not ndjson_lines:
            return

        payload = ("\n".join(ndjson_lines) + "\n").encode("utf-8")
        r = es.post(
            "/_bulk",
            body_bytes=payload,
            headers={"Content-Type": "application/x-ndjson"},
            ok_codes=(200,),
            retry=2,
            retry_sleep_s=3.0,
        )

        # 解析 bulk items：忽略 409 conflict（create 已存在）
        errors = r.get("errors", False)
        items = r.get("items", []) or []
        bad_samples = []

        for it in items:
            # it: {"create": {...}} or {"index": {...}}
            op, detail = next(iter(it.items()))
            status = int(detail.get("status", 0))
            if 200 <= status < 300:
                n_ok += 1
            elif status == 409 and resume:
                n_conflict += 1
            else:
                bad_samples.append(it)
                if len(bad_samples) >= 3:
                    break

        if errors and bad_samples:
            raise RuntimeError(f"[BULK] errors=true; sample items={bad_samples}")

        ndjson_lines.clear()

    for doc_id, text in iter_corpus_docs(corpus_path, blacklist):
        # 你要的“可见 pipeline”格式：pipeline 写在 action meta 里（Bulk API 支持）：
        # {"create": {"_index": "...", "_id": "...", "pipeline": "mtrag-elser-v1"}}
        op = "create" if resume else "index"
        ndjson_lines.append(json_dumps({op: {"_index": index_name, "_id": doc_id, "pipeline": pipeline_id}}))
        ndjson_lines.append(json_dumps({"document_id": doc_id, "text": text}))

        n_sent += 1
        if n_sent % batch_docs == 0:
            flush()
            if n_sent % (batch_docs * 10) == 0:
                dt = time.time() - t0
                rate = n_ok / max(dt, 1e-6)
                print(f"[BULK] sent={n_sent} ok={n_ok} conflict={n_conflict} rate_ok={rate:.2f} docs/s")

    flush()

    es.post(f"/{urllib.parse.quote(index_name)}/_refresh", ok_codes=(200,))
    if fast_bulk:
        set_refresh_interval(es, index_name, "1s")

    dt = time.time() - t0
    print(f"[BULK] DONE {index_name}: sent={n_sent} ok={n_ok} conflict={n_conflict} time={dt:.1f}s ok_rate={n_ok / max(dt,1e-6):.2f} docs/s")

def elser_search(es: ESClient, index_name: str, model_id: str, query_text: str, top_k: int) -> List[dict]:
    body = {
        "size": top_k,
        "_source": False,
        "query": {
            "text_expansion": {
                "ml.tokens": {
                    "model_id": model_id,
                    "model_text": query_text,
                }
            }
        },
    }
    r = es.request("GET", f"/{urllib.parse.quote(index_name)}/_search", body_obj=body, ok_codes=(200,))
    hits = r.get("hits", {}).get("hits", [])
    return [{"document_id": h.get("_id"), "score": float(h.get("_score", 0.0))} for h in hits]

def run_official_eval(input_file: str, output_file: str, model_name: str, task_name: str):
    cmd = [
        "python3", "scripts/evaluation/run_retrieval_eval.py",
        "--input_file", input_file,
        "--output_file", output_file,
        "--model_name", model_name,
        "--task_name", task_name,
    ]
    print("[EVAL] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[EVAL] Done ->", output_file)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--es_url", type=str, default="http://127.0.0.1:9200")
    ap.add_argument("--task", type=str, default="lastturn", choices=["lastturn", "questions", "rewrite"])
    ap.add_argument("--model_id", type=str, default=".elser_model_1")
    ap.add_argument("--deployment_id", type=str, default="for_search")
    ap.add_argument("--pipeline_id", type=str, default="mtrag-elser-v1")
    ap.add_argument("--index_prefix", type=str, default="mtrag")
    ap.add_argument("--index_tag", type=str, default="")
    ap.add_argument("--domains", type=str, default="clapnq,fiqa,govt,cloud")
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--no_trial", action="store_true")
    ap.add_argument("--no_index", action="store_true")
    ap.add_argument("--bulk_batch_docs", type=int, default=500)

    ap.add_argument("--corpus_override_dir", type=str, default=None)
    ap.add_argument("--corpus_override_suffix", type=str, default=".jsonl")
    ap.add_argument("--blacklist_path", type=str, default=None)

    ap.add_argument("--model_name", type=str, default="elser_v1_es810")
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--do_eval", action="store_true")

    # sleep-safe / resume knobs
    ap.add_argument("--no_resume", action="store_true", help="disable resume(create); use index (will redo)")
    ap.add_argument("--fast_bulk", action="store_true", help="set refresh_interval=-1 during bulk, restore after")
    ap.add_argument("--allocations", type=int, default=0, help="try update deployment allocations (best-effort)")

    args = ap.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    if not domains:
        raise ValueError("Empty --domains")

    es = ESClient(args.es_url, timeout_s=300)

    info = es.get("/")
    ver = (info.get("version", {}) or {}).get("number", "unknown")
    print(f"[ES] connected: version={ver} url={args.es_url}")

    if not args.no_trial:
        ensure_trial(es)

    ensure_elser_v1_model(es, args.model_id)
    ensure_deployment_started(es, args.model_id, args.deployment_id)
    if args.allocations:
        update_deployment_allocations(es, args.model_id, args.allocations)

    ensure_pipeline(es, args.pipeline_id, args.model_id)

    COLLECTIONS = {
        "clapnq": {"collection_name": "mt-rag-clapnq-elser-512-100-20240503", "root": "human/retrieval_tasks/clapnq", "corpus_file": "clapnq.jsonl", "query_file": f"clapnq_{args.task}.jsonl"},
        "fiqa":   {"collection_name": "mt-rag-fiqa-beir-elser-512-100-20240501", "root": "human/retrieval_tasks/fiqa",   "corpus_file": "fiqa.jsonl",   "query_file": f"fiqa_{args.task}.jsonl"},
        "govt":   {"collection_name": "mt-rag-govt-elser-512-100-20240611",     "root": "human/retrieval_tasks/govt",   "corpus_file": "govt.jsonl",   "query_file": f"govt_{args.task}.jsonl"},
        "cloud":  {"collection_name": "mt-rag-ibmcloud-elser-512-100-20240502", "root": "human/retrieval_tasks/cloud",  "corpus_file": "cloud.jsonl",  "query_file": f"cloud_{args.task}.jsonl"},
    }

    blacklist = load_blacklist(args.blacklist_path)

    def index_name(domain: str) -> str:
        base = f"{args.index_prefix}_{domain}"
        return f"{base}__{args.index_tag}" if args.index_tag else base

    # indexing
    for d in domains:
        idx = index_name(d)
        ensure_index(es, idx, args.pipeline_id, recreate=args.recreate)

        if not args.no_index:
            cfg = COLLECTIONS[d]
            corpus_path = os.path.join(args.corpus_override_dir, f"{d}{args.corpus_override_suffix}") if args.corpus_override_dir else os.path.join(cfg["root"], cfg["corpus_file"])
            print(f"[CORPUS] {d}: {corpus_path}")

            bulk_index(
                es=es,
                index_name=idx,
                pipeline_id=args.pipeline_id,
                corpus_path=corpus_path,
                blacklist=blacklist,
                batch_docs=args.bulk_batch_docs,
                resume=(not args.no_resume) and (not args.recreate),
                fast_bulk=args.fast_bulk,
            )

    # retrieval
    Path("outputs").mkdir(exist_ok=True)
    out_path = f"outputs/{args.model_name}_{args.task}.jsonl"
    scored_path = f"outputs/{args.model_name}_{args.task}_score.jsonl"

    all_results = []
    for d in domains:
        cfg = COLLECTIONS[d]
        q_path = os.path.join(cfg["root"], cfg["query_file"])
        queries = load_queries(q_path)
        idx = index_name(d)
        print(f"[SEARCH] domain={d} index={idx} queries={len(queries)}")
        for qid, qtext in queries.items():
            ctxs = elser_search(es, idx, args.model_id, qtext, top_k=args.top_k)
            all_results.append({"task_id": qid, "contexts": ctxs, "Collection": cfg["collection_name"]})

    with open(out_path, "w", encoding="utf-8") as f:
        for x in all_results:
            f.write(json_dumps(x) + "\n")
    print("[SAVED]", out_path)

    if args.do_eval:
        run_official_eval(out_path, scored_path, model_name=args.model_name, task_name=args.task)

if __name__ == "__main__":
    main()
