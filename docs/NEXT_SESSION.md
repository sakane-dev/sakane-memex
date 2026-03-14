# sakane-memex 次セッション引き継ぎ書
> 作成: 2026-03-05 / 引き継ぎ元スレッド終了時点

---

## 現在の状態

| 項目 | 状態 |
|------|------|
| GitHub | https://github.com/sakane-dev/sakane-memex (master) |
| Embeddingモデル | `qwen3-embedding:latest` (7.6B/Q4_K_M) — settings.yaml設定済み |
| LLMモデル | `qwen3:latest` |
| ChromaDB | **空（削除済み）→ 再インジェスト必要** |
| コード | 全修正済み・push済み |
| 仮想環境 | `D:\sakane-memex\.venv` (Python 3.12) |

---

## Step 1: インジェスト実行（最優先）

```powershell
cd D:\sakane-memex
.venv\Scripts\activate
python -m cli.corpus_cli ingest data\raw\2025-2026_mybuzzwords.md --force
python -m cli.corpus_cli stats
```

**期待値**: `Total chunks: 619` / エラーなし

---

## Step 2: 検索品質検証

```powershell
python -m cli.corpus_cli search "形式知"
python -m cli.corpus_cli search "監視資本主義"
python -m cli.corpus_cli search "CTEM"
python -m cli.corpus_cli search "境界合理性"
```

**期待値**:
- 完全一致クエリ → スコア1.000
- 意味的近傍 → 0.7以上で関連概念が返る
- `paraphrase-multilingual` 時の実績: 監視資本主義→ステークホルダー資本主義(0.760)、境界合理性→限定合理性(0.939)

---

## Step 3: `graph-build` コマンドの実装（コード追加）

### 背景
- `knowledge_graph.py` は完全実装済み
- `corpus_cli.py` に `graph-build` コマンドが**未実装**
- 新規コマンドを追加する必要がある

### 実装仕様

`cli/corpus_cli.py` に以下を追加：

```python
@cli.command("graph-build")
@click.option("--output", default="data/exports/knowledge_graph.json")
@click.option("--output-graphml", default="data/exports/knowledge_graph.graphml")
@click.option("--resume", is_flag=True, help="中断済みの進捗から再開")
@click.option("--batch-size", default=10, help="LLMへの同時投入数")
def graph_build(output, output_graphml, resume, batch_size):
    """619件全チャンクをqwen3で分析し知識グラフを構築する"""
```

### 処理フロー
1. ChromaDBから全619チャンクの `anchor_text` を取得
2. 進捗ファイル `data/exports/graph_progress.json` を確認（resume対応）
3. 未処理チャンクを `context_extractor.py` で順次LLM分析
4. `knowledge_graph.add_analysis()` で逐次グラフ構築
5. 10件ごとに進捗を `graph_progress.json` に保存（クラッシュ対策）
6. 完了後 `knowledge_graph.export()` でJSON + GraphML出力

### 実行コマンド（実装後）
```powershell
# 初回実行（数時間かかる）
python -m cli.corpus_cli graph-build

# クラッシュ後の再開
python -m cli.corpus_cli graph-build --resume
```

---

## Step 4: 知識グラフバッチ処理（夜間実行推奨）

```powershell
python -m cli.corpus_cli graph-build --output data\exports\knowledge_graph.json
```

**期待値**:
- 619件 × qwen3分析（1件あたり約5-10秒 → 合計1-2時間）
- `data/exports/knowledge_graph.json` 生成
- `data/exports/knowledge_graph.graphml` 生成

---

## Step 5: グラフ検証

```powershell
# 上位概念確認（CLIコマンドは未実装、Pythonで直接実行）
python -c "
import json
data = json.load(open('data/exports/knowledge_graph.json', encoding='utf-8'))
print('Nodes:', len(data['nodes']))
print('Edges:', len(data['edges']))
print('Paradigms:', data['stats']['paradigm_distribution'])
# 上位10概念
nodes = sorted(data['nodes'], key=lambda x: x.get('frequency',0), reverse=True)
for n in nodes[:10]:
    print(n['id'], '-', n.get('frequency',0))
"
```

---

## Step 6: API起動確認

```powershell
uvicorn src.api.main:app --reload --port 8000
```

ブラウザで `http://localhost:8000/docs` を確認。

---

## 重要な技術メモ

### chunker.py の設定
```python
# corpus_cli.py の正しい設定（settings.yamlは参照されない）
chunker = SemanticChunker(strategy="entry", context_window=0)
```
- `context_window=0` は必須（qwen3-embeddingの制約ではなくNaN対策として確定）
- `settings.yaml` の `strategy: semantic` は旧設定の残骸（無視してよい）

### NaN問題の解決済み経緯
- `nomic-embed-text` → 日本語単語でスコア1.000病
- `bge-m3` → 特定エントリでNaN（500エラー）
- `paraphrase-multilingual` → 安定だがコンテキスト長400-500文字制限
- **`qwen3-embedding`** → 採用確定（日英多言語・32768トークン・qwen3ファミリー整合）

### ファイル構成（GitHubと同期済み）
```
D:\sakane-memex\
├── cli\corpus_cli.py          # graph-buildコマンド追加が必要
├── config\settings.yaml       # qwen3-embedding設定済み
├── src\ingestor\chunker.py    # CognitiveJournalChunker
├── src\store\chroma_store.py  # NaNフィルタ付き・import math修正済み
├── src\analyzer\context_extractor.py
├── src\graph\knowledge_graph.py  # 完全実装済み
└── data\raw\2025-2026_mybuzzwords.md  # 619エントリ（コーパス本体）
```

---

## 将来タスク（Step 6以降）

- Web UI（React + FastAPI）
- 差分インジェスト（ライブドキュメント対応）
- 時系列分析（思考の変遷追跡）
- 他データソース対応（PDF・対話履歴）
