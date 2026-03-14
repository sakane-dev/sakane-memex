"""
sakane-memex / src/api/main.py

FastAPI による REST API インターフェース。

エンドポイント:
  POST /ingest          - MDファイルをインジェスト
  GET  /search          - 意味検索
  POST /analyze         - チャンクの文脈分析
  GET  /graph/concept   - 概念の知識グラフ近傍
  GET  /stats           - コーパス統計
"""

from __future__ import annotations

import logging
import json as _json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import load_config, setup_logging
from src.ingestor.chunker import SemanticChunker
from src.ingestor.notion_parser import NotionMarkdownParser
from src.store.chroma_store import CorpusStore
from src.analyzer.context_extractor import ContextExtractor

cfg = load_config()
setup_logging(cfg)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="sakane-memex API",
    description="Personal Semantic Knowledge Base — Sakane Structural Corpus",
    version="2026.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# シングルトン初期化
_store: CorpusStore | None = None
_extractor: ContextExtractor | None = None


def get_store() -> CorpusStore:
    global _store
    if _store is None:
        _store = CorpusStore(
            persist_dir=cfg["paths"]["chroma_db"],
            collection_name=cfg["store"]["collection_name"],
            ollama_base_url=cfg["embedder"]["ollama_base_url"],
            embedding_model=cfg["embedder"]["model"],
        )
    return _store


def get_extractor() -> ContextExtractor:
    global _extractor
    if _extractor is None:
        _extractor = ContextExtractor(cfg)
    return _extractor


# ------------------------------------------------------------------
# Request / Response models
# ------------------------------------------------------------------


class IngestRequest(BaseModel):
    path: str  # ファイルまたはディレクトリのパス
    recursive: bool = True
    force_reingest: bool = False  # 既存データを上書き


class SearchRequest(BaseModel):
    query: str
    n_results: int = 10
    language: str | None = None


class AnalyzeRequest(BaseModel):
    text: str
    chunk_id: str = ""


class IngestEntryRequest(BaseModel):
    """
    Web UI から直接エントリを追加するリクエスト。

    単一エントリ:
        {"entries": ["アテンション・エコノミー"]}

    複数一括（改行区切りで貼り付けた内容をそのまま渡す）:
        {"entries": ["概念A", "概念B", "概念C"]}
    """
    entries: list[str]  # 1件でもリストで渡す（UIで改行split済み）


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@app.get("/api/info")
async def root():
    return {
        "name": "sakane-memex",
        "version": "2026.1",
        "corpus_size": get_store().count(),
    }


@app.post("/ingest")
async def ingest(req: IngestRequest) -> dict[str, Any]:
    """
    指定パスのMDファイルをコーパスにインジェスト。
    Notionエクスポート構造を自動解析。
    """
    path = Path(req.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")

    parser = NotionMarkdownParser()
    chunker = SemanticChunker(
        max_chars=cfg["chunker"]["chunk_size"],
        min_chars=cfg["chunker"]["min_chunk_length"],
    )
    store = get_store()

    # ファイル or ディレクトリ
    if path.is_file():
        docs = [d for d in [parser.parse_file(path)] if d]
    else:
        docs = parser.parse_directory(path, recursive=req.recursive)

    if not docs:
        return {
            "status": "warning",
            "message": "No documents parsed",
            "chunks_added": 0,
        }

    # 再インジェスト時は既存データを削除
    if req.force_reingest:
        for doc in docs:
            store.delete_by_source(doc.source_path)

    chunks = chunker.chunk_documents(docs)
    added = store.add_chunks(chunks)

    return {
        "status": "success",
        "documents_parsed": len(docs),
        "chunks_added": added,
        "total_corpus_size": store.count(),
    }


@app.get("/search")
async def search(
    q: str = Query(..., description="検索クエリ（日本語・英語どちらも可）"),
    n: int = Query(10, ge=1, le=50),
    lang: str | None = Query(None, description="言語フィルタ: ja / en / mixed"),
) -> dict[str, Any]:
    """意味検索。クエリに意味的に近いチャンクを返す。"""
    store = get_store()
    if store.count() == 0:
        return {"results": [], "message": "Corpus is empty. Run /ingest first."}

    results = store.search(query=q, n_results=n, language=lang)
    return {
        "query": q,
        "total_results": len(results),
        "results": results,
    }


@app.post("/analyze")
async def analyze(req: AnalyzeRequest) -> dict[str, Any]:
    """
    テキストチャンクの意味論的文脈を抽出・仮説化。
    坂根構造コーパスの核心機能。
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    extractor = get_extractor()
    analysis = extractor.extract(req.text, chunk_id=req.chunk_id)

    return {
        "chunk_id": analysis.chunk_id,
        "paradigm": analysis.paradigm,
        "context_summary": analysis.context_summary,
        "key_concepts": analysis.key_concepts,
        "implicit_assumptions": analysis.implicit_assumptions,
        "semantic_hypothesis": analysis.semantic_hypothesis,
        "relations": analysis.relations,
        "temporal_marker": analysis.temporal_marker,
        "provider_used": analysis.provider_used,
    }


@app.get("/graph/concept")
async def graph_concept(
    concept: str = Query(..., description="検索する概念"),
    depth: int = Query(2, ge=1, le=3),
) -> dict[str, Any]:
    """
    指定概念の知識グラフ近傍を返す。
    坂根専用：意味的近傍から「思考の軌跡」を辿る。
    """
    # KnowledgeGraph はインメモリのため、検索結果から動的構築
    # 将来: 永続化グラフから読み込み
    store = get_store()
    results = store.search(concept, n_results=20)

    if not results:
        return {"found": False, "concept": concept, "neighbors": []}

    # 結果から関連概念を集約
    related_docs = [r["metadata"].get("doc_title", "") for r in results]
    related_scores = {r["metadata"].get("doc_title", ""): r["score"] for r in results}

    return {
        "found": True,
        "concept": concept,
        "semantic_neighbors": [
            {
                "chunk_id": r["chunk_id"],
                "doc_title": r["metadata"].get("doc_title", ""),
                "score": r["score"],
                "heading_context": r["metadata"].get("heading_context", ""),
                "language": r["metadata"].get("language", ""),
                "text_preview": r["text"][:200] + "..."
                if len(r["text"]) > 200
                else r["text"],
            }
            for r in results
        ],
    }


@app.get("/stats")
async def stats() -> dict[str, Any]:
    """コーパス統計情報を返す。"""
    store = get_store()
    sources = store.list_sources()
    return {
        "total_chunks": store.count(),
        "total_sources": len(sources),
        "sources": sources,
    }


@app.get("/graph/data")
async def graph_data() -> dict[str, Any]:
    """knowledge_graph.jsonをそのまま返す。"""
    graph_path = Path("data/exports/knowledge_graph.json")
    if not graph_path.exists():
        raise HTTPException(status_code=404, detail="Knowledge graph not found. Run graph-build first.")
    with open(graph_path, encoding="utf-8") as f:
        return _json.load(f)


@app.get("/graph/top")
async def graph_top(n: int = Query(20, ge=5, le=100)) -> dict[str, Any]:
    """上位n件の概念とエッジを返す（グラフ・ヒートマップ用）。"""
    graph_path = Path("data/exports/knowledge_graph.json")
    if not graph_path.exists():
        raise HTTPException(status_code=404, detail="Knowledge graph not found.")
    with open(graph_path, encoding="utf-8") as f:
        data = _json.load(f)
    nodes = sorted(data["nodes"], key=lambda x: x.get("frequency", 0), reverse=True)[:n]
    top_ids = {nd["id"] for nd in nodes}
    edges = [e for e in data["edges"] if e["source"] in top_ids and e["target"] in top_ids]
    return {
        "nodes": nodes,
        "edges": edges,
        "stats": data["stats"],
    }


@app.get("/umap/data")
async def umap_data() -> dict[str, Any]:
    """UMAP投影データを返す（埋め込み空間可視化用）。"""
    umap_path = Path("data/exports/umap_projection.json")
    if not umap_path.exists():
        raise HTTPException(status_code=404, detail="UMAP data not found. Run scripts/build_umap.py first.")
    with open(umap_path, encoding="utf-8") as f:
        return _json.load(f)


@app.post("/ingest/entry")
async def ingest_entry(req: IngestEntryRequest) -> dict[str, Any]:
    """
    Web UI から直接エントリをコーパスに追加する。

    単一・複数一括どちらも受け付ける。
    chunk_id は anchor_text のみから生成するため、
    MDファイル経由のインジェストと衝突しない（同一テキストは同一ID）。

    空行・空文字列は自動的にスキップされる。
    """
    from src.ingestor.chunker import make_chunk_from_text
    from datetime import datetime, timezone

    # 空行を除去して有効エントリのみ処理
    valid_entries = [e.strip() for e in req.entries if e.strip()]
    if not valid_entries:
        raise HTTPException(status_code=400, detail="有効なエントリが1件もありません")

    store = get_store()
    chunks = [make_chunk_from_text(text=entry) for entry in valid_entries]
    added = store.add_chunks(chunks)

    # ingest_state.json は MD フロー専用のため Web UI 入力では更新しない
    # （entry_number=0 のエントリは時系列分析では除外される設計）

    return {
        "status": "success",
        "submitted": len(valid_entries),
        "chunks_added": added,
        "total_corpus_size": store.count(),
        "entries": valid_entries,
    }


# フロントエンドの静的ファイルをマウント（ルーティングの最後に配置する）
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
