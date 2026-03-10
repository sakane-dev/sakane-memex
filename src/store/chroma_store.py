"""
sakane-memex / src/store/chroma_store.py

ChromaDB による永続化ベクトルストア。
Ollama qwen3-embedding を embedding 関数として使用。
"""

from __future__ import annotations

import logging
import math
from typing import Any

import chromadb
from chromadb import Collection
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from src.ingestor.chunker import Chunk

logger = logging.getLogger(__name__)


class CorpusStore:
    """
    坂根構造コーパスのベクトルストア。

    使用例:
        store = CorpusStore(persist_dir="data/chroma_db")
        store.add_chunks(chunks)
        results = store.search("形式知と暗黙知の対立", n_results=5)
    """

    def __init__(
        self,
        persist_dir: str = "data/chroma_db",
        collection_name: str = "sakane_corpus",
        ollama_base_url: str = "http://localhost:11434",
        embedding_model: str = "qwen3-embedding:latest",
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # ChromaDB クライアント（永続化）
        self._client = chromadb.PersistentClient(path=persist_dir)

        # Ollama embedding 関数
        self._embed_fn = OllamaEmbeddingFunction(
            url=f"{ollama_base_url}/api/embeddings",
            model_name=embedding_model,
        )

        # コレクション取得または作成
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "CorpusStore initialized: collection='%s', items=%d",
            collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 32) -> int:
        """
        チャンクをベクトルDBに追加。
        既存の chunk_id は上書き（upsert）。
        返り値: 追加/更新件数
        """
        if not chunks:
            return 0

        added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            # バッチ内を1件ずつ処理してNaN/500エラーを個別スキップ
            for c in batch:
                try:
                    emb_raw = self._embed_fn([c.text])
                    emb = emb_raw[0]
                    emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                    if not emb_list or any(math.isnan(v) for v in emb_list):
                        logger.warning(
                            "Skipped NaN embedding: entry=%s text=%s",
                            c.metadata.get("entry_number", "?"),
                            c.anchor_text[:40]
                            if hasattr(c, "anchor_text")
                            else c.text[:40],
                        )
                        continue
                    self._collection.upsert(
                        ids=[c.chunk_id],
                        documents=[c.text],
                        metadatas=[self._sanitize_metadata(c.metadata)],
                        embeddings=[emb_list],
                    )
                    added += 1
                except Exception as e:
                    logger.warning(
                        "Skipped chunk entry=%s: %s",
                        c.metadata.get("entry_number", "?"),
                        str(e)[:80],
                    )

        logger.info("Added/updated %d chunks to '%s'", added, self.collection_name)
        return added

    def delete_by_source(self, source_path: str) -> int:
        """指定ソースパスのチャンクを全削除。再インジェスト時に使用。"""
        try:
            results = self._collection.get(
                where={"source_path": source_path},
                include=[],
            )
            ids = results.get("ids", [])
            if ids:
                self._collection.delete(ids=ids)
                logger.info("Deleted %d chunks from source: %s", len(ids), source_path)
            return len(ids)
        except Exception as e:
            logger.error("Failed to delete chunks for %s: %s", source_path, e)
            return 0

    # ------------------------------------------------------------------
    # Read / Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        n_results: int = 10,
        where: dict | None = None,
        language: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        意味検索。クエリに意味的に近いチャンクを返す。

        Args:
            query: 検索クエリ（日本語・英語どちらも可）
            n_results: 返す件数
            where: メタデータフィルタ（例: {"language": "ja"}）
            language: 言語フィルタのショートカット

        Returns:
            [{chunk_id, text, score, metadata}, ...]
        """
        filters = where or {}
        if language:
            filters["language"] = language

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(n_results, self._collection.count() or 1),
                where=filters if filters else None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error("Search failed: %s", e)
            return []

        hits = []
        for i, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            score = 1.0 - distance  # cosine distance → similarity
            hits.append(
                {
                    "chunk_id": doc_id,
                    "text": results["documents"][0][i],
                    "score": round(score, 4),
                    "metadata": results["metadatas"][0][i],
                }
            )

        return hits

    def get_by_id(self, chunk_id: str) -> dict[str, Any] | None:
        """IDでチャンクを取得。"""
        try:
            result = self._collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"],
            )
            if result["ids"]:
                return {
                    "chunk_id": result["ids"][0],
                    "text": result["documents"][0],
                    "metadata": result["metadatas"][0],
                }
        except Exception as e:
            logger.error("get_by_id failed for %s: %s", chunk_id, e)
        return None

    def get_all_chunks(self) -> list[dict[str, Any]]:
        """全チャンクをIDとメタデータ付きで返す（graph-build用）。"""
        try:
            result = self._collection.get(
                include=["documents", "metadatas"],
            )
            chunks = []
            for i, chunk_id in enumerate(result["ids"]):
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": result["documents"][i],
                        "metadata": result["metadatas"][i],
                    }
                )
            return chunks
        except Exception as e:
            logger.error("get_all_chunks failed: %s", e)
            return []

    def count(self) -> int:
        """格納済みチャンク数を返す。"""
        return self._collection.count()

    def list_sources(self) -> list[str]:
        """インジェスト済みのソースパス一覧を返す。"""
        try:
            results = self._collection.get(include=["metadatas"])
            paths = {m.get("source_path", "") for m in results.get("metadatas", [])}
            return sorted(filter(None, paths))
        except Exception as e:
            logger.error("list_sources failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """
        ChromaDB は str/int/float/bool のみ受け付けるため
        list や None を変換する。
        """
        sanitized = {}
        for k, v in metadata.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            elif isinstance(v, list):
                sanitized[k] = ", ".join(str(x) for x in v)
            else:
                sanitized[k] = str(v)
        return sanitized
