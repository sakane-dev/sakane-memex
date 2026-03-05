"""
sakane-memex / src/graph/knowledge_graph.py

概念間の意味的関係を NetworkX グラフとして構築・可視化する。

ノード: 概念（key_concepts から抽出）
エッジ: LLMが抽出した relations + 共起関係
属性: パラダイム, 出現頻度, 意味的仮説
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

from src.analyzer.context_extractor import ContextAnalysis

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    坂根構造コーパスの知識グラフ。

    使用例:
        kg = KnowledgeGraph()
        kg.build_from_analyses(analyses)
        kg.export("data/exports/knowledge_graph.json")
    """

    def __init__(self, min_edge_weight: int = 2):
        self.min_edge_weight = min_edge_weight
        self.G: nx.Graph = nx.Graph()
        self._paradigm_counter: Counter = Counter()
        self._concept_sources: defaultdict = defaultdict(list)

    def build_from_analyses(self, analyses: list[ContextAnalysis]) -> None:
        """ContextAnalysis リストからグラフを構築。"""
        logger.info("Building knowledge graph from %d analyses", len(analyses))

        co_occurrence: Counter = Counter()
        explicit_relations: list[tuple[str, str, str]] = []

        for analysis in analyses:
            concepts = [c.lower().strip() for c in analysis.key_concepts if c.strip()]

            # ノードの追加・属性更新
            for concept in concepts:
                if not self.G.has_node(concept):
                    self.G.add_node(
                        concept,
                        frequency=0,
                        paradigms=[],
                        hypotheses=[],
                        sources=[],
                    )
                self.G.nodes[concept]["frequency"] += 1
                if analysis.paradigm:
                    self.G.nodes[concept]["paradigms"].append(analysis.paradigm)
                if analysis.semantic_hypothesis:
                    self.G.nodes[concept]["hypotheses"].append(
                        analysis.semantic_hypothesis
                    )
                self.G.nodes[concept]["sources"].append(analysis.chunk_id)
                self._concept_sources[concept].append(analysis.source_path
                    if hasattr(analysis, 'source_path') else "")

            # 共起ペアのカウント
            for i, c1 in enumerate(concepts):
                for c2 in concepts[i + 1 :]:
                    key = tuple(sorted([c1, c2]))
                    co_occurrence[key] += 1

            # LLM抽出の明示的関係
            for rel in analysis.relations:
                src = str(rel.get("from", "")).lower().strip()
                tgt = str(rel.get("to", "")).lower().strip()
                rel_type = str(rel.get("relation", "related")).strip()
                if src and tgt:
                    explicit_relations.append((src, tgt, rel_type))

            if analysis.paradigm:
                self._paradigm_counter[analysis.paradigm] += 1

        # 共起エッジの追加
        for (c1, c2), weight in co_occurrence.items():
            if weight >= self.min_edge_weight:
                if self.G.has_edge(c1, c2):
                    self.G[c1][c2]["weight"] += weight
                else:
                    self.G.add_edge(c1, c2, weight=weight, relation_type="co-occurrence")

        # 明示的関係エッジの追加
        for src, tgt, rel_type in explicit_relations:
            if not self.G.has_node(src):
                self.G.add_node(src, frequency=1, paradigms=[], hypotheses=[], sources=[])
            if not self.G.has_node(tgt):
                self.G.add_node(tgt, frequency=1, paradigms=[], hypotheses=[], sources=[])
            if self.G.has_edge(src, tgt):
                self.G[src][tgt]["weight"] += 1
                existing = self.G[src][tgt].get("relation_type", "")
                if rel_type not in existing:
                    self.G[src][tgt]["relation_type"] = f"{existing}, {rel_type}"
            else:
                self.G.add_edge(src, tgt, weight=1, relation_type=rel_type)

        logger.info(
            "Graph built: %d nodes, %d edges",
            self.G.number_of_nodes(),
            self.G.number_of_edges(),
        )

    def get_top_concepts(self, n: int = 20) -> list[dict[str, Any]]:
        """出現頻度・次数が高い上位概念を返す。"""
        nodes = []
        for node, data in self.G.nodes(data=True):
            degree = self.G.degree(node)
            freq = data.get("frequency", 0)
            paradigms = list(set(data.get("paradigms", [])))
            nodes.append({
                "concept": node,
                "frequency": freq,
                "degree": degree,
                "centrality_score": freq * degree,
                "paradigms": paradigms,
                "hypotheses": data.get("hypotheses", [])[:2],
            })
        return sorted(nodes, key=lambda x: x["centrality_score"], reverse=True)[:n]

    def search_concept(self, concept: str, depth: int = 2) -> dict[str, Any]:
        """指定概念の近傍グラフを返す（坂根専用：意味的近傍探索）。"""
        concept = concept.lower().strip()
        if not self.G.has_node(concept):
            # 部分一致検索
            matches = [n for n in self.G.nodes if concept in n]
            if not matches:
                return {"found": False, "concept": concept}
            concept = matches[0]

        neighbors = list(nx.ego_graph(self.G, concept, radius=depth).nodes(data=True))
        edges = list(nx.ego_graph(self.G, concept, radius=depth).edges(data=True))

        return {
            "found": True,
            "concept": concept,
            "frequency": self.G.nodes[concept].get("frequency", 0),
            "paradigms": list(set(self.G.nodes[concept].get("paradigms", []))),
            "hypotheses": self.G.nodes[concept].get("hypotheses", []),
            "neighbors": [
                {
                    "concept": n,
                    "frequency": d.get("frequency", 0),
                    "edge_weight": self.G[concept][n]["weight"]
                    if self.G.has_edge(concept, n)
                    else 0,
                    "relation": self.G[concept][n].get("relation_type", "")
                    if self.G.has_edge(concept, n)
                    else "",
                }
                for n, d in neighbors
                if n != concept
            ],
            "paradigm_distribution": dict(self._paradigm_counter),
        }

    def export(self, output_path: str, format: str = "json") -> None:
        """グラフをファイルにエクスポート。"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "graphml":
            # NetworkX の GraphML エクスポート（リスト型属性を文字列に変換）
            G_export = self.G.copy()
            for node, data in G_export.nodes(data=True):
                for key in ("paradigms", "hypotheses", "sources"):
                    if key in data and isinstance(data[key], list):
                        G_export.nodes[node][key] = " | ".join(str(x) for x in data[key])
            nx.write_graphml(G_export, str(path))

        elif format == "json":
            data = {
                "nodes": [
                    {
                        "id": n,
                        **{
                            k: (v if not isinstance(v, list) else v)
                            for k, v in d.items()
                        },
                    }
                    for n, d in self.G.nodes(data=True)
                ],
                "edges": [
                    {"source": u, "target": v, **d}
                    for u, v, d in self.G.edges(data=True)
                ],
                "stats": {
                    "total_nodes": self.G.number_of_nodes(),
                    "total_edges": self.G.number_of_edges(),
                    "paradigm_distribution": dict(self._paradigm_counter),
                },
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("Graph exported to %s (%s)", path, format)
