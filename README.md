# sakane-memex v2026.1

> "When Insight Becomes Structure"

坂根康之の認知ジャーナル（`2025-2026_mybuzzwords.md`、619エントリ）を知識コーパスとして扱い、意味・文脈・パラダイムを抽出・グラフ化・可視化するパーソナル意味論的ナレッジベース。

---

## 設計思想

**前提：** 人間の思考は線形ではない。バズワードリストは「脳内DBの外部化」であり、制約なしの認知ストリームである。このシステムはその認知ストリームに構造を与え、思考の軌跡を可視化することを目的とする。

**核心原則：**

- **1エントリー = 1チャンク** — 番号付きリストの1行が認知の最小粒度。分割しない。
- **`context_window=0`** — qwen3-embeddingはチャンク単体でembedding。NaN対策として確定。
- **`settings.yaml`のchunker.strategyは参照されない** — 実動作は`corpus_cli.py`の`SemanticChunker(strategy="entry", context_window=0)`が全て。
- **オントロジー必須** — 知識グラフ化は不変の設計原則。ベクトル検索だけでは意味階層を捉えられない。
- **ローカルファースト** — 全処理をRTX3060/RAM64GBのローカル環境で完結。APIキー不要（Geminiはfallbackのみ）。

---

## システムアーキテクチャ

```
[ data/raw/*.md ]
      │
      ▼
[ NotionMarkdownParser ]       ← Notion番号プレフィックス保持
      │  parse_file()
      ▼
[ SemanticChunker ]            ← strategy="entry", context_window=0
      │  chunk_documents()      　1エントリー = 1 Chunk オブジェクト
      ▼
[ CorpusStore (ChromaDB) ]     ← OllamaEmbeddingFunction(qwen3-embedding:latest)
      │  add_chunks()            　NaNフィルタ付きupsert、cosine距離
      │  search()                　意味検索
      │  get_all_embeddings()    　UMAP用ベクトル全件取得
      ▼
[ ContextExtractor ]           ← qwen3:latest (primary) / Gemini (fallback)
      │  extract()               　パラダイム分類・意味論的仮説生成
      │  extract_batch()         　619件バッチ処理
      ▼
[ KnowledgeGraph (NetworkX) ]  ← ノード=key_concepts、エッジ=共起+明示的関係
      │  build_from_analyses()
      │  export()                　JSON + GraphML
      ▼
[ UMAP Projection ]            ← 4096次元 → 2次元（cosineメトリクス）
      │  scripts/build_umap.py
      ▼
[ FastAPI + Web UI ]           ← 5機能SPA（検索・グラフ・ヒートマップ・分析・Embedding Space）
```

---

## 技術スタック

| 層 | 技術 | 備考 |
|---|---|---|
| Embedding | `qwen3-embedding:latest` (7.6B/Q4_K_M) | 日英多言語・32768トークン対応 |
| LLM | `qwen3:latest` | パラダイム分類・仮説生成・知識グラフ構築 |
| LLM Fallback | Gemini 2.0 Flash | `GEMINI_API_KEY`環境変数から取得 |
| Vector Store | ChromaDB (PersistentClient) | cosine距離・NaNフィルタ付きupsert |
| 次元削減 | UMAP (`umap-learn`) | 4096dim→2dim・metric=cosine・random_state=42 |
| 知識グラフ | NetworkX | JSON + GraphML エクスポート |
| API | FastAPI + uvicorn | CORS対応・StaticFilesマウント |
| CLI | Click + Rich | ingest / search / analyze / stats / graph-build |
| フロントエンド | Vanilla JS + D3.js v7 | ビルドツール不要・single HTML file |

**ハードウェア：** Windows 11 / RTX3060 12GB / RAM 64GB / Python 3.12

---

## embedding モデル選定経緯

| モデル | 結果 | 理由 |
|---|---|---|
| `nomic-embed-text` | ❌ | 日本語単語でスコア1.000病 |
| `bge-m3` | ❌ | 特定エントリでNaN（500エラー） |
| `paraphrase-multilingual` | △ | 安定だがコンテキスト長400-500文字制限 |
| **`qwen3-embedding:latest`** | ✅ | 日英多言語・32768トークン・qwen3ファミリー整合 |

---

## ディレクトリ構成

```
D:\sakane-memex\
├── cli\
│   └── corpus_cli.py          # CLIエントリポイント（5コマンド）
├── config\
│   ├── _config.py             # load_config, setup_logging
│   └── settings.yaml          # 設定（chunker.strategyは参照されない点に注意）
├── data\
│   ├── raw\
│   │   └── 2025-2026_mybuzzwords.md   # コーパス本体（619エントリ）
│   ├── chroma_db\             # ChromaDB永続化（自動生成）
│   └── exports\
│       ├── knowledge_graph.json       # 知識グラフ（1759ノード・1257エッジ）
│       ├── knowledge_graph.graphml    # GraphML形式
│       ├── umap_projection.json       # UMAP2次元投影（619点）
│       └── graph_progress.json        # LLM分析進捗（resume対応）
├── frontend\
│   └── index.html             # Web UI（5タブSPA）
├── scripts\
│   └── build_umap.py          # UMAP投影生成スクリプト
├── src\
│   ├── ingestor\
│   │   ├── notion_parser.py   # Notion MD対応パーサー
│   │   └── chunker.py         # CognitiveJournalChunker（1エントリー=1チャンク）
│   ├── store\
│   │   └── chroma_store.py    # ChromaDB・NaNフィルタ・get_all_embeddings()
│   ├── analyzer\
│   │   └── context_extractor.py   # LLM文脈抽出（ContextAnalysis dataclass）
│   ├── graph\
│   │   └── knowledge_graph.py     # NetworkX知識グラフ
│   └── api\
│       └── main.py            # FastAPI（7エンドポイント）
└── tests\
```

---

## セットアップ

### 前提条件

```powershell
# Ollama起動確認
ollama serve

# 必須モデル
ollama pull qwen3-embedding:latest
ollama pull qwen3:latest
```

### 環境構築

```powershell
cd D:\sakane-memex
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 環境変数

```powershell
@"
GEMINI_API_KEY=your_key_here
OLLAMA_BASE_URL=http://localhost:11434
"@ | Out-File -FilePath .env -Encoding utf8
```

---

## 基本操作

```powershell
cd D:\sakane-memex
.venv\Scripts\Activate.ps1

# Step 1: インジェスト（ChromaDB空の場合）
python -m cli.corpus_cli ingest data\raw\2025-2026_mybuzzwords.md
python -m cli.corpus_cli stats
# 期待値: Total chunks: 619

# Step 2: 検索品質検証
python -m cli.corpus_cli search "形式知"
python -m cli.corpus_cli search "監視資本主義"
python -m cli.corpus_cli search "CTEM"
python -m cli.corpus_cli search "境界合理性"

# Step 3: 知識グラフ構築（619件 × qwen3分析、約2.5時間）
python -m cli.corpus_cli graph-build
# クラッシュ後の再開
python -m cli.corpus_cli graph-build --resume

# Step 4: UMAP投影生成
python scripts/build_umap.py
# パラメータ調整例
python scripts/build_umap.py --n-neighbors 20 --min-dist 0.05

# Step 5: API + Web UI起動
uvicorn src.api.main:app --reload --port 8000
# ブラウザ: http://localhost:8000
```

---

## CLI コマンドリファレンス

```powershell
# ingest: MDファイルをコーパスにインジェスト
python -m cli.corpus_cli ingest <path> [--force]
# --force: delete_by_source()でソース単位削除（ChromaDB全体クリアではない）

# search: 意味検索
python -m cli.corpus_cli search "<query>" [-n 10] [--lang ja]

# analyze: 単一テキストのLLM分析
python -m cli.corpus_cli analyze --text "<text>"
python -m cli.corpus_cli analyze --file <path>

# stats: コーパス統計
python -m cli.corpus_cli stats

# graph-build: 全619件をqwen3で分析し知識グラフ構築
python -m cli.corpus_cli graph-build [--resume] [--batch-size 10]
```

---

## API エンドポイント

| Method | Path | 説明 |
|---|---|---|
| GET | `/api/info` | バージョン・コーパスサイズ |
| POST | `/ingest` | MDファイルインジェスト |
| GET | `/search?q=&n=` | 意味検索 |
| POST | `/analyze` | テキスト文脈分析 |
| GET | `/graph/concept?concept=` | 概念の意味的近傍 |
| GET | `/graph/top?n=` | 上位N概念・エッジ（グラフ・ヒートマップ用） |
| GET | `/graph/data` | knowledge_graph.json全体 |
| GET | `/umap/data` | UMAP投影データ（619点） |
| GET | `/stats` | コーパス統計 |

---

## Web UI 機能

`http://localhost:8000` でアクセス。5タブ構成のSPA（Vanilla JS + D3.js）。

| タブ | 機能 |
|---|---|
| `// search` | 意味検索・スコアバー可視化 |
| `// graph` | Force-directed graph（ノードサイズ=頻度・色=パラダイム）|
| `// heatmap` | 上位N概念間の共起強度マトリクス |
| `// analyze` | 任意テキストのリアルタイムLLM分析 |
| `// stats` | パラダイム分布バー・コーパス統計 |
| `// embedding` | qwen3-embedding空間のUMAP2次元投影（619点・色=パラダイム・サイズ=概念豊富さ）|

**UMAP解釈注意：** 近接点の意味的類似性は高精度で保証されるが、離れた点同士の非類似性は保証されない（4096次元→2次元の情報損失）。

---

## 知識グラフ仕様

- **ノード:** 1,759個（LLMが抽出したkey_concepts）
- **エッジ:** 1,257個（共起 + 明示的関係）
- **最強結合:** 認知プロセス ↔ 言語構造（weight=7）

**パラダイム分布:**

| パラダイム | 件数 | 比率 |
|---|---|---|
| 技術戦略的 | 297 | 48% |
| 認知科学的 | 189 | 31% |
| 哲学的 | 100 | 16% |
| 社会工学的 | 21 | 3% |
| 存在論的 | 6 | 1% |

**コーパスの三層構造:**
- 基底層（哲学的）: 言語・存在・意味論がメタフレームとして機能
- 媒介層（認知科学的）: 認知プロセスが哲学と技術をつなぐ翻訳装置
- 表層（技術戦略的）: クラウド・NLP・MLが応用層として位置づけ

---

## 重要な実装メモ

### chunker.py
```python
# corpus_cli.pyの正しい設定（settings.yamlは参照されない）
chunker = SemanticChunker(strategy="entry", context_window=0)
# context_window=0 は必須（qwen3-embeddingの制約ではなくNaN対策として確定）
```

### chroma_store.py
- `add_chunks()`: 1件ずつ処理してNaN/500エラーを個別スキップ
- `delete_by_source()`: `--force`フラグで呼ばれる。ChromaDB全体クリアではなくソース単位削除
- `get_all_embeddings()`: UMAP用。`include=["documents","metadatas","embeddings"]`で全件取得

### graph-build コマンド
- 進捗を`data/exports/graph_progress.json`に10件ごと自動保存
- `--resume`で中断再開可能
- qwen3の`<think>`タグをregexで除去してJSON解析

### build_umap.py
- ChromaDB → 4096次元ベクトル全件取得 → UMAP(cosine) → 正規化[0,1] → JSON保存
- `graph_progress.json`からパラダイム情報をchunk_idでマッピング
- frequencyは各チャンクのkey_concepts数（LLMが抽出した概念数）を代理指標として使用

---

## ロードマップ

- [x] qwen3-embedding によるインジェスト（619チャンク）
- [x] 検索品質検証（監視資本主義→デジタルパノプティコン 0.775等）
- [x] graph-build CLIコマンド実装
- [x] 知識グラフ構築（1759ノード・1257エッジ）
- [x] UMAP埋め込み空間可視化
- [x] Web UI（5タブSPA）
- [x] FastAPI（9エンドポイント）
- [ ] 近傍点動的ハイライト（Embedding Spaceタブ）
- [ ] 差分インジェスト（ライブドキュメント対応）
- [ ] 時系列分析（思考の変遷追跡）
- [ ] 他データソース対応（PDF・対話履歴）
- [ ] Web UI（React化・Vite）
