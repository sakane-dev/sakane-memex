# sakane-memex v2026.3

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
- **ソース非依存インジェスト** — MDファイル経由・Web UI直接入力の両方に対応。`chunk_id`はanchor_textのみから生成するため経路間で衝突しない。

---

## システムアーキテクチャ

```
【入力経路A】 [ data/raw/*.md ]
                    │
                    ▼
      [ NotionMarkdownParser ]       ← Notion番号プレフィックス保持
                    │  parse_file()
                    ▼
      [ SemanticChunker ]            ← strategy="entry", context_window=0
                    │  chunk_documents()  1エントリー = 1 Chunk オブジェクト
                    ▼
【入力経路B】 [ /ingest/entry API ]  ← Web UIから直接入力（単一・バルク）
                    │  make_chunk_from_text()
                    ▼
      [ CorpusStore (ChromaDB) ]     ← OllamaEmbeddingFunction(qwen3-embedding:latest)
                    │  add_chunks()        NaNフィルタ付きupsert、cosine距離
                    │  search()            意味検索
                    │  get_all_embeddings() UMAP用ベクトル全件取得
                    ▼
      [ ContextExtractor ]           ← qwen3:latest (primary) / Gemini (fallback)
                    │  extract()           パラダイム分類・意味論的仮説生成
                    │  extract_batch()     619件バッチ処理
                    ▼
      [ KnowledgeGraph (NetworkX) ]  ← ノード=key_concepts、エッジ=共起+明示的関係
                    │  build_from_analyses()
                    │  export()            JSON + GraphML
                    ▼
      [ UMAP Projection ]            ← 4096次元 → 2次元（cosineメトリクス）
                    │  scripts/build_umap.py
                    ▼
      [ FastAPI + Web UI ]           ← 7タブSPA（検索・グラフ・ヒートマップ・分析・統計・Embedding Space・Add Entry）
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
│       ├── graph_progress.json        # LLM分析進捗（resume対応）
│       └── ingest_state.json          # 差分インジェスト状態管理（Phase 2追加）
├── frontend\
│   └── index.html             # Web UI（7タブSPA）
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
│       └── main.py            # FastAPI（10エンドポイント）
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

# Step 1: 初回インジェスト（全件）
python -m cli.corpus_cli ingest data\raw\2025-2026_mybuzzwords.md
python -m cli.corpus_cli stats
# 期待値: Total chunks: 619

# Step 2: 差分インジェスト（Notionに追記後）
python -m cli.corpus_cli ingest data\raw\2025-2026_mybuzzwords.md --incremental
# → data/exports/ingest_state.json を参照して新規エントリのみ処理

# Step 3: 検索品質検証
python -m cli.corpus_cli search "形式知"
python -m cli.corpus_cli search "監視資本主義"
python -m cli.corpus_cli search "CTEM"
python -m cli.corpus_cli search "境界合理性"

# Step 4: 知識グラフ構築（619件 × qwen3分析、約2.5時間）
python -m cli.corpus_cli graph-build
# クラッシュ後の再開
python -m cli.corpus_cli graph-build --resume
# 差分インジェスト後の追加分のみ分析
python -m cli.corpus_cli graph-build --incremental

# Step 5: UMAP投影生成
python scripts/build_umap.py
# パラメータ調整例
python scripts/build_umap.py --n-neighbors 20 --min-dist 0.05

# Step 6: API + Web UI起動
uvicorn src.api.main:app --reload --port 8000
# ブラウザ: http://localhost:8000
```

---

## CLI コマンドリファレンス

```powershell
# ingest: MDファイルをコーパスにインジェスト
python -m cli.corpus_cli ingest <path>
python -m cli.corpus_cli ingest <path> --incremental   # 差分のみ（追記専用運用）
python -m cli.corpus_cli ingest <path> --force         # ソース単位削除後に全件再インジェスト
# ※ --force と --incremental の同時使用は不可

# search: 意味検索
python -m cli.corpus_cli search "<query>" [-n 10] [--lang ja]

# analyze: 単一テキストのLLM分析
python -m cli.corpus_cli analyze --text "<text>"
python -m cli.corpus_cli analyze --file <path>

# stats: コーパス統計
python -m cli.corpus_cli stats

# graph-build: 全件をqwen3で分析し知識グラフ構築
python -m cli.corpus_cli graph-build [--resume] [--incremental] [--batch-size 10]
```

---

## API エンドポイント

| Method | Path | 説明 |
|---|---|---|
| GET | `/api/info` | バージョン・コーパスサイズ |
| POST | `/ingest` | MDファイルインジェスト |
| POST | `/ingest/entry` | Web UIから直接エントリ追加（単一・バルク） |
| GET | `/search?q=&n=` | 意味検索 |
| POST | `/analyze` | テキスト文脈分析 |
| GET | `/graph/concept?concept=` | 概念の意味的近傍 |
| GET | `/graph/top?n=` | 上位N概念・エッジ（グラフ・ヒートマップ用） |
| GET | `/graph/data` | knowledge_graph.json全体 |
| GET | `/umap/data` | UMAP投影データ（619点） |
| GET | `/stats` | コーパス統計 |

---

## Web UI 機能

`http://localhost:8000` でアクセス。7タブ構成のSPA（Vanilla JS + D3.js）。

| タブ | 機能 |
|---|---|
| `// search` | 意味検索・スコアバー可視化 |
| `// graph` | Force-directed graph（ノードサイズ=頻度・色=パラダイム）|
| `// heatmap` | 上位N概念間の共起強度マトリクス |
| `// analyze` | 任意テキストのリアルタイムLLM分析 |
| `// stats` | パラダイム分布バー・コーパス統計 |
| `// embedding` | qwen3-embedding空間のUMAP2次元投影（619点・色=パラダイム・サイズ=概念豊富さ）|
| `// add` | コーパスへの直接エントリ追加（Single / Bulkモード切替）|

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

### chunker.py（Phase 2改修済み）
```python
# corpus_cli.pyの正しい設定（settings.yamlは参照されない）
chunker = SemanticChunker(strategy="entry", context_window=0)
# context_window=0 は必須（qwen3-embeddingの制約ではなくNaN対策として確定）

# chunk_id生成ロジック（Phase 2変更）
# 旧: SHA256(source_path + entry_number + text[:128])
# 新: SHA256(anchor_text[:256])
# → MDファイル経由・Web UI直接入力の両経路で衝突しない
# → 同一テキストは常に同一IDになりupsertが冪等
```

### chroma_store.py
- `add_chunks()`: 1件ずつ処理してNaN/500エラーを個別スキップ
- `delete_by_source()`: `--force`フラグで呼ばれる。ChromaDB全体クリアではなくソース単位削除
- `get_all_embeddings()`: UMAP用。`include=["documents","metadatas","embeddings"]`で全件取得

### corpus_cli.py（Phase 2改修済み）
- `ingest --incremental`: `ingest_state.json`の`last_entry_number`を基準に新規エントリのみ処理
- `ingest --force`: `delete_by_source()`後に全件再インジェスト、`ingest_state.json`もリセット
- `graph-build --incremental`: `graph_progress.json`の最大`entry_number`を基準に新規チャンクのみLLM分析
- `graph_progress.json`の`analyses`に`entry_number`フィールドを追加（incremental判定の基準）

### ingest_state.json（Phase 2新規追加）
```json
{
  "last_entry_number": 619,
  "last_ingested_at": "2026-03-14T00:00:00+00:00",
  "total_chunks": 619
}
```
- 保存先: `data/exports/ingest_state.json`
- MDフロー専用。Web UI入力（`entry_number=0`）は更新しない
- `entry_number=0`のエントリは時系列分析（Phase 3）では除外される設計

### main.py（Phase 2改修済み）
- `/ingest/entry`エンドポイント追加
- `IngestEntryRequest`: `entries: list[str]`（1件でもリストで渡す）
- `make_chunk_from_text()`を使用してWeb UI入力をChromaDBに直接upsert

### graph-build コマンド
- 進捗を`data/exports/graph_progress.json`に10件ごと自動保存
- `--resume`で中断再開可能
- qwen3の`<think>`タグをregexで除去してJSON解析

### build_umap.py
- ChromaDB → 4096次元ベクトル全件取得 → UMAP(cosine) → 正規化[0,1] → JSON保存
- `graph_progress.json`からパラダイム情報をchunk_idでマッピング
- frequencyは各チャンクのkey_concepts数（LLMが抽出した概念数）を代理指標として使用

---

## 差分インジェスト運用フロー（Phase 2）

```
【Notionに新規buzzwordを追記した後】

1. Notion → MDエクスポート → data\raw\2025-2026_mybuzzwords.md を上書き

2. 差分インジェスト
   python -m cli.corpus_cli ingest data\raw\2025-2026_mybuzzwords.md --incremental
   # → ingest_state.jsonのlast_entry_numberより大きいエントリのみChromaDBに追加

3. 知識グラフ更新（新規エントリのみLLM分析）
   python -m cli.corpus_cli graph-build --incremental
   # → graph_progress.jsonの最大entry_numberより大きいエントリのみ処理
   # → グラフ全体を再構築してknowledge_graph.jsonを上書き

4. UMAP再生成（全件ベクトルから再投影）
   python scripts/build_umap.py

【Web UIから直接追加する場合】
   → // add タブ → Single（1ワード）またはBulk（改行区切り複数）で入力 → Add to Corpus
   → ChromaDBに即時upsert・corpus badge更新
   → entry_number=0として記録（時系列分析では除外）
   → 後からNotionノートに手動追記することを推奨（バックアップ）
```

---

## ロードマップ

### 完了済み（v2026.1）
- [x] qwen3-embedding によるインジェスト（619チャンク）
- [x] 検索品質検証（監視資本主義→デジタルパノプティコン 0.775等）
- [x] graph-build CLIコマンド実装（resume対応）
- [x] 知識グラフ構築（1759ノード・1257エッジ）
- [x] UMAP埋め込み空間可視化（4096dim→2dim）
- [x] Web UI 6タブSPA（search / graph / heatmap / analyze / stats / embedding）
- [x] FastAPI 9エンドポイント
- [x] memex.bat 起動管理ツール

### 完了済み（v2026.3 / Phase 2）
- [x] 差分インジェスト（`ingest --incremental` / `ingest_state.json`）
- [x] `graph-build --incremental`（新規チャンクのみLLM分析）
- [x] Web UI `// add` タブ（Single / Bulkモード切替）
- [x] `/ingest/entry` APIエンドポイント
- [x] `chunk_id`生成のソース非依存化（MDファイル・Web UI経路の衝突防止）

### Phase 3: 時系列分析（中期）
- [ ] エントリー番号を時間軸として解釈（1→619が時系列順）
- [ ] パラダイムの変遷タイムライン可視化（D3.js streamgraph）
- [ ] Web UIに `// timeline` タブ追加
- [ ] `/timeline/data` エンドポイント追加
- [ ] ※ `entry_number=0`（Web UI入力）は時系列分析から除外

### Phase 4: 抽象度×体化度の対軸実装（中期・SaaS設計直結）
- [ ] `scripts/build_axes.py` 新規作成（qwen3にabstraction_score・embodiment_scoreを問う）
- [ ] Lakoffの概念メタファー理論を参照プロンプトとして使用
- [ ] `graph_progress.json`に2スコアフィールドを追加
- [ ] Web UIに `// axes` タブ追加（X軸=抽象度・Y軸=体化度の散布図）
- [ ] `/axes/data` エンドポイント追加

**4象限モデル:**

| 象限 | 抽象度 | 体化度 | 概念例 | AIの危険度 |
|------|--------|--------|--------|-----------|
| Ⅰ 浮遊する言語 | 高 | 低 | 差延・論理形式 | 最高: 幻覚が起きやすい |
| Ⅱ 生きた抽象 | 高 | 高 | 愛・信頼・責任 | 最危険: 言えているように見えるが言えていない |
| Ⅲ 硬い地形 | 低 | 低 | アルゴリズム・仕様 | 最低: AIが最も信頼できる領域 |
| Ⅳ 体化した具体 | 低 | 高 | 痛み・疲労・境界 | 中: 身体経験がないため欠落 |

### Phase 5: インフラ整備（随時）
- [ ] requirements.txt 更新（umap-learn・scikit-learn 追加）
- [ ] 一時スクリプト削除（ルートに残存するretry_failed*.py等）
- [ ] `graph-build`のJSON解析失敗率改善（`format='json'`オプション使用）
- [ ] chunk_id変更に伴う既存619チャンクの再インジェスト（`ingest --force`）

### Phase 6: SaaS化準備（将来）
- [ ] フロントエンド: Next.js 15 + React 19 + R3F（Three.js）による3D Embedding Space
- [ ] バックエンド: 商用LLM API（Claude/GPT/Gemini）をラップ
- [ ] インフラ: Vercel + Supabase + Clerk
- [ ] モデル戦略: LoRA/QLoRAでファインチューニングしたqwen3-7Bベースモデル
