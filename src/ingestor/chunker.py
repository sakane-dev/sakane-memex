"""
sakane-memex / src/ingestor/chunker.py

知識コーパス対応チャンカー。

設計思想:
  2025-2026_mybuzzwords.md = 知識コーパス
  各エントリー = コーパスの1トークン

  1語・1フレーズのembeddingは意味空間が機能しない。
  前後N件のエントリーをコンテキストウィンドウとして付加し、
  分布意味論的なembeddingを実現する。

  例:
    entry 71: "損失回避バイアス・現状維持バイアス・ハーディング効果"
    entry 72: "アテンション・エコノミー"  ← anchor
    entry 73: "Blameless Post-mortems"
    → embedding input:
      "損失回避バイアス・現状維持バイアス・ハーディング効果
       アテンション・エコノミー
       Blameless Post-mortems"
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from src.ingestor.notion_parser import ParsedDocument

logger = logging.getLogger(__name__)

ChunkStrategy = Literal["entry", "paragraph", "line"]

_NUMBERED_ENTRY = re.compile(r"^\s*(\d+)\.\s*(.*)$")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_IMAGE_LINE = re.compile(r"^!\[.*\]\(.*\)$")


@dataclass
class Chunk:
    chunk_id: str
    text: str  # コンテキストウィンドウ付きテキスト（embedding用）
    anchor_text: str  # アンカーエントリー本体（検索結果表示用）
    raw_text: str  # 元テキスト
    source_path: str
    doc_title: str
    entry_number: int
    chunk_index: int
    total_chunks: int
    metadata: dict[str, Any] = field(default_factory=dict)
    language: str = "unknown"
    char_count: int = 0
    has_url: bool = False
    is_long_form: bool = False


def _make_chunk_id(source_path: str, entry_number: int, text: str) -> str:
    seed = f"{source_path}::{entry_number}::{text[:128]}"
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


def _clean_entry_text(raw: str) -> str:
    text = _MD_LINK.sub(r"\1", raw)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _detect_language(text: str) -> str:
    ja = len(re.findall(r"[\u3040-\u9fff]", text))
    ratio = ja / max(len(text), 1)
    if ratio > 0.15:
        return "ja"
    if ratio > 0.03:
        return "mixed"
    return "en"


def _is_long_form(text: str) -> bool:
    has_punct = bool(re.search(r"[。、．，.,:：]", text))
    return len(text) >= 40 and has_punct


class CognitiveJournalChunker:
    """
    知識コーパス専用チャンカー。
    各エントリーを前後N件のコンテキストウィンドウ付きでembeddingする。
    """

    def __init__(
        self,
        strategy: ChunkStrategy = "entry",
        skip_empty: bool = True,
        min_chars: int = 1,
        context_window: int = 2,  # 前後何件を文脈として付加するか
        # 後方互換パラメータ
        max_chars: int = 0,
        chunk_overlap: int = 0,
        chunk_size: int = 0,
        overlap_chars: int = 0,
    ):
        self.strategy = strategy
        self.skip_empty = skip_empty
        self.min_chars = min_chars
        self.context_window = context_window

    def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
        if not doc.raw_text.strip():
            logger.warning("Empty document: %s", doc.source_path)
            return []

        if self.strategy == "entry":
            raw_entries = self._extract_numbered_entries(doc.raw_text)
        elif self.strategy == "paragraph":
            raw_entries = self._extract_paragraphs(doc.raw_text)
        else:
            raw_entries = self._extract_lines(doc.raw_text)

        # フィルタリング
        valid_entries: list[tuple[int, str]] = []
        for entry_num, raw_text in raw_entries:
            if self.skip_empty and not raw_text.strip():
                continue
            if _IMAGE_LINE.match(raw_text.strip()):
                continue
            clean = _clean_entry_text(raw_text)
            if len(clean) < self.min_chars:
                continue
            valid_entries.append((entry_num, raw_text))

        total = len(valid_entries)
        # クリーンテキストのリスト（コンテキストウィンドウ参照用）
        clean_texts = [_clean_entry_text(rt) for _, rt in valid_entries]

        chunks = []
        for idx, (entry_num, raw_text) in enumerate(valid_entries):
            anchor = _clean_entry_text(raw_text)

            # コンテキストウィンドウ: 前後N件を付加
            w = self.context_window
            start = max(0, idx - w)
            end = min(total, idx + w + 1)
            context_texts = clean_texts[start:end]
            embedding_text = "\n".join(context_texts)

            has_url = bool(re.search(r"https?://", raw_text))
            long_form = _is_long_form(anchor)
            lang = _detect_language(anchor)
            chunk_id = _make_chunk_id(doc.source_path, entry_num, anchor)

            chunk = Chunk(
                chunk_id=chunk_id,
                text=embedding_text,  # embedding用（文脈付き）
                anchor_text=anchor,  # 表示用（エントリー本体）
                raw_text=raw_text.strip(),
                source_path=doc.source_path,
                doc_title=doc.title,
                entry_number=entry_num,
                chunk_index=idx,
                total_chunks=total,
                language=lang,
                char_count=len(anchor),
                has_url=has_url,
                is_long_form=long_form,
                metadata={
                    "source_path": doc.source_path,
                    "doc_title": doc.title,
                    "entry_number": entry_num,
                    "chunk_index": idx,
                    "total_chunks": total,
                    "language": lang,
                    "has_url": has_url,
                    "is_long_form": long_form,
                    "char_count": len(anchor),
                    "anchor_text": anchor,  # 検索結果で表示するテキスト
                    "tags": ", ".join(doc.tags) if doc.tags else "",
                    "notion_id": doc.notion_id,
                },
            )
            chunks.append(chunk)

        logger.info(
            "Chunked '%s' -> %d entry chunks (context_window=%d)",
            doc.title,
            len(chunks),
            self.context_window,
        )
        return chunks

    def chunk_documents(self, docs: list[ParsedDocument]) -> list[Chunk]:
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        logger.info("Total: %d chunks from %d documents", len(all_chunks), len(docs))
        return all_chunks

    def _extract_numbered_entries(self, text: str) -> list[tuple[int, str]]:
        entries: list[tuple[int, str]] = []
        for line in text.split("\n"):
            m = _NUMBERED_ENTRY.match(line)
            if m:
                num = int(m.group(1))
                content = m.group(2).strip()
                entries.append((num, content))
        logger.debug("Extracted %d numbered entries", len(entries))
        return entries

    def _extract_paragraphs(self, text: str) -> list[tuple[int, str]]:
        paragraphs = re.split(r"\n\n+", text)
        return [(i + 1, p.strip()) for i, p in enumerate(paragraphs) if p.strip()]

    def _extract_lines(self, text: str) -> list[tuple[int, str]]:
        return [
            (i + 1, line.strip())
            for i, line in enumerate(text.split("\n"))
            if line.strip()
        ]


# 後方互換エイリアス
SemanticChunker = CognitiveJournalChunker
