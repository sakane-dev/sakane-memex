"""
sakane-memex / src/ingestor/notion_parser.py

Notionエクスポート（Markdown形式）専用パーサー。
Notion固有の構造を解析し、構造化ドキュメントオブジェクトに変換する。

Notion MDの特徴:
  - ファイル名にブロックIDが付与される例: "My Note abc1234def5.md"
  - frontmatter は存在しないことが多い
  - ネストされたページは サブディレクトリ + index.md 構造
  - プロパティブロック（Notion DB由来）が先頭に存在する場合がある
  - タグは "#tag" 形式または Notion プロパティ "Tags: ..." として存在
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import frontmatter

logger = logging.getLogger(__name__)

# NotionエクスポートのブロックID（末尾32文字hex）を除去するパターン
_NOTION_ID_PATTERN = re.compile(r"\s+[0-9a-f]{32}$", re.IGNORECASE)
_NOTION_ID_SHORT = re.compile(r"\s+[0-9a-f]{8,32}$", re.IGNORECASE)

# Notion プロパティ行（例: "Tags: AI, LLM"）
_PROPERTY_LINE = re.compile(r"^([A-Za-z\u30a0-\u9fff]+):\s*(.+)$")


@dataclass
class ParsedDocument:
    """パース済みドキュメントの構造化表現。"""
    source_path: str
    title: str
    raw_text: str                          # クリーニング後テキスト
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    properties: dict[str, str] = field(default_factory=dict)
    headings: list[dict[str, Any]] = field(default_factory=list)  # {level, text, position}
    language: str = "unknown"              # "ja" | "en" | "mixed"
    char_count: int = 0
    notion_id: str = ""                    # ファイル名から抽出したNotion block ID


class NotionMarkdownParser:
    """
    Notionエクスポートされたマークダウンファイルを解析する。

    使用例:
        parser = NotionMarkdownParser()
        docs = parser.parse_directory("data/raw")
        for doc in docs:
            print(doc.title, len(doc.raw_text))
    """

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_file(self, path: Path | str) -> ParsedDocument | None:
        """単一MDファイルを解析してParseDocument を返す。"""
        path = Path(path)
        if not path.exists():
            logger.warning("File not found: %s", path)
            return None
        if path.suffix.lower() not in (".md", ".txt"):
            logger.debug("Skipping non-markdown file: %s", path)
            return None

        try:
            raw_content = path.read_text(encoding=self.encoding, errors="replace")
        except Exception as e:
            logger.error("Failed to read %s: %s", path, e)
            return None

        return self._parse_content(raw_content, source_path=str(path))

    def parse_directory(
        self,
        directory: Path | str,
        recursive: bool = True,
    ) -> list[ParsedDocument]:
        """
        ディレクトリ内の全MDファイルを解析する。
        Notionエクスポートのサブディレクトリ構造も再帰的に処理。
        """
        directory = Path(directory)
        if not directory.exists():
            logger.error("Directory not found: %s", directory)
            return []

        pattern = "**/*.md" if recursive else "*.md"
        files = sorted(directory.glob(pattern))
        logger.info("Found %d markdown files in %s", len(files), directory)

        docs = []
        for f in files:
            doc = self.parse_file(f)
            if doc and doc.char_count >= 10:  # 極端に短いファイルをスキップ
                docs.append(doc)

        logger.info("Successfully parsed %d documents", len(docs))
        return docs

    # ------------------------------------------------------------------
    # Internal parsing logic
    # ------------------------------------------------------------------

    def _parse_content(self, content: str, source_path: str) -> ParsedDocument:
        path = Path(source_path)

        # 1. Notion ブロックID をファイル名から抽出
        notion_id = self._extract_notion_id(path.stem)
        title = self._clean_title(path.stem)

        # 2. frontmatter の解析（存在する場合）
        try:
            post = frontmatter.loads(content)
            fm_metadata = dict(post.metadata)
            body = post.content
        except Exception:
            fm_metadata = {}
            body = content

        # 3. Notion プロパティブロックの解析（frontmatter外のプロパティ）
        properties, body = self._extract_notion_properties(body)

        # 4. タグの収集（frontmatter + プロパティ + インラインタグ）
        tags = self._collect_tags(fm_metadata, properties, body)

        # 5. 見出し構造の抽出
        headings = self._extract_headings(body)

        # 6. タイトル補完（H1見出しがあればそれをタイトルに）
        if headings and headings[0]["level"] == 1:
            title = headings[0]["text"]

        # 7. テキストクリーニング
        clean_text = self._clean_text(body)

        # 8. 言語推定
        language = self._detect_language(clean_text)

        # メタデータ統合
        metadata = {
            **fm_metadata,
            "source_path": source_path,
            "notion_id": notion_id,
            "title": title,
            "tags": tags,
            "language": language,
            **{f"prop_{k}": v for k, v in properties.items()},
        }

        return ParsedDocument(
            source_path=source_path,
            title=title,
            raw_text=clean_text,
            metadata=metadata,
            tags=tags,
            properties=properties,
            headings=headings,
            language=language,
            char_count=len(clean_text),
            notion_id=notion_id,
        )

    def _extract_notion_id(self, stem: str) -> str:
        """ファイル名末尾のNotion ブロックIDを抽出。"""
        m = _NOTION_ID_SHORT.search(stem)
        return m.group(0).strip() if m else ""

    def _clean_title(self, stem: str) -> str:
        """ファイル名からNotion IDと特殊文字を除去してタイトルを生成。"""
        title = _NOTION_ID_SHORT.sub("", stem).strip()
        title = re.sub(r"[_\-]+", " ", title).strip()
        return title

    def _extract_notion_properties(
        self, body: str
    ) -> tuple[dict[str, str], str]:
        """
        Notion DB由来のプロパティブロックを先頭から抽出。
        例:
            Tags: AI, LLM, 構造知性
            Status: Active
            Created: 2024-01-15
        """
        lines = body.split("\n")
        properties: dict[str, str] = {}
        end_idx = 0

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                if i > 0 and properties:
                    # プロパティブロック終了
                    end_idx = i
                    break
                continue
            m = _PROPERTY_LINE.match(stripped)
            if m and i < 20:  # 先頭20行のみプロパティとして解釈
                key = m.group(1).lower().strip()
                val = m.group(2).strip()
                properties[key] = val
                end_idx = i + 1
            elif i > 0 and properties:
                end_idx = i
                break

        remaining_body = "\n".join(lines[end_idx:]).strip()
        return properties, remaining_body

    def _collect_tags(
        self,
        fm_metadata: dict,
        properties: dict,
        body: str,
    ) -> list[str]:
        """frontmatter、プロパティ、インラインタグを統合収集。"""
        tags: set[str] = set()

        # frontmatter の tags/tag フィールド
        for key in ("tags", "tag", "Tags", "Tag"):
            val = fm_metadata.get(key)
            if val:
                if isinstance(val, list):
                    tags.update(str(t).strip().lower() for t in val)
                else:
                    tags.update(t.strip().lower() for t in str(val).split(","))

        # Notion プロパティの tags
        for key in ("tags", "tag", "category", "categories"):
            if key in properties:
                tags.update(
                    t.strip().lower() for t in properties[key].split(",")
                )

        # インラインタグ (#tag 形式)
        inline = re.findall(r"(?<!\w)#([\w\u3040-\u9fff\-]+)", body)
        tags.update(t.lower() for t in inline)

        return sorted(filter(None, tags))

    def _extract_headings(self, body: str) -> list[dict[str, Any]]:
        """マークダウン見出しを抽出。"""
        headings = []
        for i, line in enumerate(body.split("\n")):
            m = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
            if m:
                headings.append({
                    "level": len(m.group(1)),
                    "text": m.group(2).strip(),
                    "line": i,
                })
        return headings

    def _clean_text(self, text: str) -> str:
        """
        マークダウン記号・HTMLタグ・Notion固有記号を除去して
        純粋なテキストを返す。
        """
        # HTMLタグ除去
        text = re.sub(r"<[^>]+>", " ", text)
        # コードブロック（内容は保持）
        text = re.sub(r"```[\w]*\n?", "", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        # 見出し記号
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # リストマーカー
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
        # 番号付きリストの番号は保持（チャンカーが使用するため除去しない）
        # 太字・斜体
        text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
        text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
        # リンク
        text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        # 水平線
        text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
        # 引用
        text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
        # Notion チェックボックス
        text = re.sub(r"^\s*-\s*\[[ x]\]\s*", "", text, flags=re.MULTILINE)
        # 余分な空白・改行
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _detect_language(self, text: str) -> str:
        """
        簡易言語判定。
        日本語文字（ひらがな・カタカナ・漢字）の比率で判断。
        """
        if not text:
            return "unknown"
        ja_chars = len(re.findall(r"[\u3040-\u9fff]", text))
        ratio = ja_chars / max(len(text), 1)
        if ratio > 0.15:
            return "ja"
        if ratio > 0.03:
            return "mixed"
        return "en"
