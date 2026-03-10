import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from umap import UMAP

# プロジェクトルートからの相対パスでsrcとconfigを解決
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config, setup_logging
from src.store.chroma_store import CorpusStore

# ロギング設定
# 存在しない場合は、基本的な設定を行う
try:
    setup_logging()
except Exception:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def build_umap_projection(
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    output_dir: Path = Path("data/exports"),
    config_path: Path = Path("config/settings.yaml"),
):
    """
    ChromaDBから全ベクトルを取得し、UMAPで2次元に投影してJSONとして保存する。
    """
    logger.info("Starting UMAP projection build...")
    logger.info(
        f"Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}"
    )

    # 設定とストアの初期化
    cfg = load_config(config_path)
    store = CorpusStore(
        persist_dir=cfg["paths"]["chroma_db"],
        collection_name=cfg["store"]["collection_name"],
        ollama_base_url=cfg["embedder"]["ollama_base_url"],
        embedding_model=cfg["embedder"]["model"],
    )

    # 1. 全ベクトルデータを取得
    logger.info("Fetching all embeddings from ChromaDB...")
    all_items = store.get_all_embeddings()
    if not all_items:
        logger.error("No embeddings found in the database. Aborting.")
        return
    logger.info(f"  Found {len(all_items)} items.")

    embeddings = np.array([item["embedding"] for item in all_items])

    # 2. UMAPで次元削減
    logger.info("Running UMAP for dimensionality reduction...")
    umap_reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=2,
        random_state=42,
        transform_queue_size=1.0,
    )
    projection = umap_reducer.fit_transform(embeddings)
    logger.info("  UMAP reduction complete.")

    # 3. 座標を[0, 1]に正規化
    scaler = MinMaxScaler()
    projection = scaler.fit_transform(projection)
    logger.info("  Projection scaled to [0, 1] range.")

    # 4. 補助情報をマージ
    # graph_progress.jsonからパラダイム情報を取得
    paradigm_map = {}
    progress_path = Path("data/exports/graph_progress.json")
    if progress_path.exists():
        try:
            progress_data = json.load(open(progress_path, encoding="utf-8"))
            for item in progress_data.get("analyses", []):
                paradigm_map[item["chunk_id"]] = item.get("paradigm", "")
            logger.info(f"  Paradigm map loaded: {len(paradigm_map)} entries")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(
                f"Could not load or parse paradigm map from {progress_path}: {e}"
            )

    # knowledge_graph.jsonからfrequency情報を取得
    # chunk_idベースでfrequencyを構築
    # 各チャンクのkey_conceptsがgraph nodesに何回登場するかの最大値を使用
    # パラダイム内での相対的重要度をfrequencyとして使用
    # graph_progress内で同じパラダイムに何件属するかをカウント
    frequency_map: dict[str, int] = {}
    progress_path2 = Path("data/exports/graph_progress.json")
    if progress_path2.exists():
        progress2 = json.load(open(progress_path2, encoding="utf-8"))
        analyses = progress2.get("analyses", [])
        # key_conceptsの総数をchunkの重要度とする
        for a in analyses:
            concepts = a.get("key_concepts", [])
            # conceptsが多い＝LLMが多くの概念を抽出した＝内容が豊富
            frequency_map[a["chunk_id"]] = max(len(concepts), 1)
        print(f"  Frequency map loaded: {len(frequency_map)} chunks")

    # 5. 出力データ構築
    logger.info("Building final JSON output...")
    points = []
    for i, item in enumerate(all_items):
        chunk_id = item["chunk_id"]
        anchor = item["metadata"].get("anchor_text", item["text"])
        paradigm = paradigm_map.get(chunk_id)

        # anchor_textからfrequencyを逆引き
        frequency = frequency_map.get(chunk_id, 1)

        points.append(
            {
                "chunk_id": chunk_id,
                "x": float(projection[i, 0]),
                "y": float(projection[i, 1]),
                "text": anchor[:120],
                "paradigm": paradigm,
                "frequency": frequency,
                "doc_title": item["metadata"].get("doc_title", ""),
                "language": item["metadata"].get("language", ""),
                "entry_number": item["metadata"].get("entry_number", ""),
            }
        )

    output_data = {
        "meta": {
            "total": len(points),
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "metric": metric,
        },
        "points": points,
    }

    # 6. JSONファイルに保存
    output_path = output_dir / "umap_projection.json"
    output_dir.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Successfully saved UMAP projection to: {output_path}")
    logger.info("Build complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build UMAP projection from ChromaDB embeddings."
    )
    parser.add_argument(
        "--n-neighbors", type=int, default=15, help="UMAP n_neighbors parameter."
    )
    parser.add_argument(
        "--min-dist", type=float, default=0.1, help="UMAP min_dist parameter."
    )
    parser.add_argument(
        "--metric", type=str, default="cosine", help="UMAP metric parameter."
    )
    args = parser.parse_args()

    build_umap_projection(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
    )
