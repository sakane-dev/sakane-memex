# sakane-memex v2026

> "When Insight Becomes Structure"

坂根の認知ジャーナル（`2025-2026_mybuzzwords.md`）を知識コーパスとして扱い、
意味・文脈・パラダイムを抽出するパーソナル意味論的ナレッジベース。

## 設計思想

- **知識コーパス**: バズワードリストは「脳内DBの外部化」。制約なしの認知ストリーム。
- **1エントリー = 1チャンク**: 番号付きリストの1行が認知の最小粒度。
- **コンテキストウィンドウ**: 前後2件を付加してembedding。分布意味論的処理。
- **オントロジー必須**: 知識グラフ化は不変の設計原則。

## 技術スタック

| 層           | 技術                                          |
| ------------ | --------------------------------------------- |
| Embedding    | `bge-m3:latest`（日英多言語対応）via Ollama   |
| LLM          | `qwen3:latest`（日本語推論）/ Gemini fallback |
| Vector Store | ChromaDB（ローカル永続化）                    |
| Graph        | NetworkX → JSON/GraphML                       |
| API          | FastAPI                                       |
| CLI          | Click + Rich                                  |

## セットアップ

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 必須: `__init__.py` 作成

```powershell
New-Item -ItemType File src\__init__.py
New-Item -ItemType File src\ingestor\__init__.py
New-Item -ItemType File src\store\__init__.py
New-Item -ItemType File src\analyzer\__init__.py
New-Item -ItemType File src\graph\__init__.py
New-Item -ItemType File src\api\__init__.py
New-Item -ItemType File cli\__init__.py
New-Item -ItemType File config\__init__.py
New-Item -ItemType File tests\__init__.py
```

### `.env` 作成

```powershell
@"
GEMINI_API_KEY=your_key_here
OLLAMA_BASE_URL=http://localhost:11434
"@ | Out-File -FilePath .env -Encoding utf8
```

### Ollamaモデル確認

```powershell
ollama list
# 必須: bge-m3:latest, qwen3:latest
```

## 使い方

```powershell
# インジェスト
python -m cli.corpus_cli ingest data\raw\ --recursive

# 検索
python -m cli.corpus_cli search "境界合理性"

# 文脈分析
python -m cli.corpus_cli analyze --text "監視資本主義"

# 統計
python -m cli.corpus_cli stats

# API起動
uvicorn src.api.main:app --reload --port 8000
```

## ディレクトリ構成

```
D:\sakane-memex
├── cli\corpus_cli.py
├── config\
│   ├── __init__.py        # load_config, setup_logging
│   └── settings.yaml
├── data\
│   ├── raw\               # Notion MDファイル（インジェスト対象）
│   ├── chroma_db\         # ベクトルDB（自動生成）
│   └── exports\           # グラフエクスポート
├── src\
│   ├── ingestor\
│   │   ├── notion_parser.py   # Notion MD対応パーサー
│   │   └── chunker.py         # 認知ジャーナル対応チャンカー
│   ├── store\
│   │   └── chroma_store.py    # ChromaDB + NaNフィルタ
│   ├── analyzer\
│   │   └── context_extractor.py  # LLM文脈抽出
│   ├── graph\
│   │   └── knowledge_graph.py    # NetworkX知識グラフ
│   └── api\main.py
└── tests\
```

## ロードマップ

- [x] 環境構築（Python 3.12 / ChromaDB / Ollama）
- [x] Notion MDパーサー（番号プレフィックス保持）
- [x] 認知ジャーナルチャンカー（1エントリー=1チャンク）
- [x] コンテキストウィンドウ付きembedding（bge-m3）
- [x] NaNフィルタ付きupsert
- [ ] 619チャンク完全格納確認
- [ ] 検索品質検証
- [ ] **知識グラフ構築・可視化**（必須）
- [ ] 差分インジェスト（ライブドキュメント対応）
- [ ] Web UI（React + FastAPI）
