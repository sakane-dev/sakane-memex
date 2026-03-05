"""
sakane-memex / src/analyzer/context_extractor.py

LLMによる意味論的文脈抽出エンジン。
坂根構造コーパスの核心部：単語の「意味の階層」と「パラダイム」を
機械に仮説化させる。

Primary:  Ollama qwen3:latest（ローカル）
Fallback: Gemini API（gemini-2.0-flash）
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ContextAnalysis:
    """
    1チャンクに対する文脈分析結果。
    """
    chunk_id: str
    chunk_text: str

    # LLMが抽出した構造
    paradigm: str = ""                     # 主要パラダイム（例: 認知科学的, 社会工学的）
    context_summary: str = ""              # 文脈要約（2-3文）
    key_concepts: list[str] = field(default_factory=list)   # 中核概念リスト
    implicit_assumptions: list[str] = field(default_factory=list)  # 暗黙の前提
    semantic_hypothesis: str = ""          # 意味論的仮説
    relations: list[dict[str, str]] = field(default_factory=list)  # 概念間関係
    temporal_marker: str = ""              # 時期・文脈マーカー

    provider_used: str = ""                # "ollama" or "gemini"
    raw_llm_response: str = ""


_EXTRACTION_PROMPT = """\
あなたは認知科学と言語哲学に精通した知識構造アナリストです。
以下のテキストを分析し、**JSON形式のみ**で応答してください。

## 分析対象テキスト
{text}

## 抽出指示
以下の項目をJSONで返してください。余分な説明や```は不要です。

{{
  "paradigm": "このテキストが属する知的パラダイム（例: 認知科学的, 社会工学的, 哲学的, 技術戦略的, 存在論的）",
  "context_summary": "このテキストが語っている文脈を2-3文で要約",
  "key_concepts": ["中核概念1", "中核概念2", "中核概念3"],
  "implicit_assumptions": ["暗黙の前提1", "暗黙の前提2"],
  "semantic_hypothesis": "このテキストで使われた言葉の意味論的位置づけに関する仮説（1文）",
  "relations": [
    {{"from": "概念A", "relation": "関係の種類", "to": "概念B"}}
  ],
  "temporal_marker": "このテキストが示す思考の時期・段階の特徴（不明な場合は空文字）"
}}
"""


class ContextExtractor:
    """
    チャンクテキストからLLMを用いて意味論的文脈を抽出する。

    使用例:
        extractor = ContextExtractor(cfg)
        analysis = extractor.extract("形式知と暗黙知の...", chunk_id="abc123")
    """

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self._analyzer_cfg = cfg.get("analyzer", {})
        self._primary = self._analyzer_cfg.get("primary", {})
        self._fallback = self._analyzer_cfg.get("fallback", {})

    def extract(self, text: str, chunk_id: str = "") -> ContextAnalysis:
        """
        テキストから文脈分析を実行。
        Primary（Ollama）失敗時はFallback（Gemini）を使用。
        """
        prompt = _EXTRACTION_PROMPT.format(text=text[:2000])  # 長文は切り詰め

        # Primary: Ollama
        result = self._call_ollama(prompt)
        if result:
            parsed = self._parse_llm_response(result)
            if parsed:
                return ContextAnalysis(
                    chunk_id=chunk_id,
                    chunk_text=text,
                    provider_used="ollama",
                    raw_llm_response=result,
                    **parsed,
                )

        # Fallback: Gemini
        logger.warning("Ollama failed, falling back to Gemini API")
        result = self._call_gemini(prompt)
        if result:
            parsed = self._parse_llm_response(result)
            if parsed:
                return ContextAnalysis(
                    chunk_id=chunk_id,
                    chunk_text=text,
                    provider_used="gemini",
                    raw_llm_response=result,
                    **parsed,
                )

        # 両方失敗した場合は空の分析を返す
        logger.error("Both LLM providers failed for chunk: %s", chunk_id)
        return ContextAnalysis(chunk_id=chunk_id, chunk_text=text, provider_used="none")

    def extract_batch(
        self,
        items: list[tuple[str, str]],  # [(chunk_id, text), ...]
        progress_callback=None,
    ) -> list[ContextAnalysis]:
        """複数チャンクをバッチ処理。"""
        results = []
        for i, (chunk_id, text) in enumerate(items):
            analysis = self.extract(text, chunk_id)
            results.append(analysis)
            if progress_callback:
                progress_callback(i + 1, len(items))
        return results

    # ------------------------------------------------------------------
    # LLM Clients
    # ------------------------------------------------------------------

    def _call_ollama(self, prompt: str) -> str | None:
        """Ollama API を呼び出す。"""
        try:
            import ollama
            model = self._primary.get("model", "qwen3:latest")
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": self._primary.get("temperature", 0.3),
                    "num_predict": self._primary.get("max_tokens", 1024),
                },
            )
            return response.get("response", "").strip()
        except Exception as e:
            logger.warning("Ollama call failed: %s", e)
            return None

    def _call_gemini(self, prompt: str) -> str | None:
        """Gemini API を呼び出す。"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not set in environment")
            return None
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model_name = self._fallback.get("model", "gemini-2.0-flash")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=self._fallback.get("temperature", 0.3),
                    max_output_tokens=self._fallback.get("max_tokens", 1024),
                ),
            )
            return response.text.strip()
        except Exception as e:
            logger.error("Gemini API call failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_llm_response(self, response: str) -> dict[str, Any] | None:
        """LLMレスポンスからJSONを抽出してパース。"""
        # コードブロックの除去
        cleaned = re.sub(r"```(?:json)?\s*", "", response)
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()

        # qwen3のthinkingタグを除去
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

        # JSON部分の抽出
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not json_match:
            logger.warning("No JSON found in LLM response")
            return None

        try:
            data = json.loads(json_match.group(0))
            return {
                "paradigm": str(data.get("paradigm", "")),
                "context_summary": str(data.get("context_summary", "")),
                "key_concepts": list(data.get("key_concepts", [])),
                "implicit_assumptions": list(data.get("implicit_assumptions", [])),
                "semantic_hypothesis": str(data.get("semantic_hypothesis", "")),
                "relations": list(data.get("relations", [])),
                "temporal_marker": str(data.get("temporal_marker", "")),
            }
        except json.JSONDecodeError as e:
            logger.warning("JSON parse failed: %s | Response: %s", e, cleaned[:200])
            return None
