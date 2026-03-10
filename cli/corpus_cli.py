"""
sakane-memex / cli/corpus_cli.py

Click CLI — コーパス操作のメインインターフェース。

使用例:
  python -m cli.corpus_cli ingest data/raw/
  python -m cli.corpus_cli search "形式知と暗黙知"
  python -m cli.corpus_cli analyze --chunk-id abc123
  python -m cli.corpus_cli stats
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config, setup_logging
from src.ingestor.notion_parser import NotionMarkdownParser
from src.ingestor.chunker import SemanticChunker
from src.store.chroma_store import CorpusStore
from src.analyzer.context_extractor import ContextExtractor

console = Console()


def _init_store(cfg: dict) -> CorpusStore:
    return CorpusStore(
        persist_dir=cfg["paths"]["chroma_db"],
        collection_name=cfg["store"]["collection_name"],
        ollama_base_url=cfg["embedder"]["ollama_base_url"],
        embedding_model=cfg["embedder"]["model"],
    )


@click.group()
@click.option("--config", default=None, help="設定ファイルパス")
@click.pass_context
def cli(ctx, config):
    """sakane-memex: パーソナル意味論的ナレッジベース"""
    ctx.ensure_object(dict)
    cfg = load_config(config)
    setup_logging(cfg)
    ctx.obj["cfg"] = cfg


# ------------------------------------------------------------------
# ingest
# ------------------------------------------------------------------


@cli.command()
@click.argument("path")
@click.option(
    "--recursive/--no-recursive", default=True, help="サブディレクトリを再帰処理"
)
@click.option("--force", is_flag=True, help="既存データを削除して再インジェスト")
@click.pass_context
def ingest(ctx, path, recursive, force):
    """MDファイルをコーパスにインジェストする。

    PATH: ファイルまたはディレクトリのパス（NotionエクスポートMD対応）
    """
    cfg = ctx.obj["cfg"]
    input_path = Path(path)

    if not input_path.exists():
        console.print(f"[red]Error: Path not found: {path}[/red]")
        sys.exit(1)

    console.print(Panel(f"[bold cyan]Ingesting: {path}[/bold cyan]"))

    parser = NotionMarkdownParser()
    chunker = SemanticChunker(strategy="entry", context_window=0)
    store = _init_store(cfg)

    with console.status("[cyan]Parsing documents...[/cyan]"):
        if input_path.is_file():
            docs = [d for d in [parser.parse_file(input_path)] if d]
        else:
            docs = parser.parse_directory(input_path, recursive=recursive)

    if not docs:
        console.print("[yellow]Warning: No documents parsed.[/yellow]")
        return

    console.print(f"[green]✓ Parsed {len(docs)} documents[/green]")

    if force:
        with console.status("[yellow]Removing existing data...[/yellow]"):
            for doc in docs:
                store.delete_by_source(doc.source_path)

    with console.status(
        f"[cyan]Chunking and embedding {len(docs)} documents...[/cyan]"
    ):
        chunks = chunker.chunk_documents(docs)
        added = store.add_chunks(chunks)

    console.print(f"[green]✓ Added {added} chunks[/green]")
    console.print(f"[dim]Total corpus size: {store.count()} chunks[/dim]")

    # サマリーテーブル
    table = Table(title="Ingestion Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Documents parsed", str(len(docs)))
    table.add_row("Chunks created", str(len(chunks)))
    table.add_row("Chunks stored", str(added))
    table.add_row("Total corpus", str(store.count()))
    console.print(table)


# ------------------------------------------------------------------
# search
# ------------------------------------------------------------------


@cli.command()
@click.argument("query")
@click.option("-n", "--results", default=5, help="返す件数")
@click.option("--lang", default=None, help="言語フィルタ: ja / en / mixed")
@click.option("--json-output", is_flag=True, help="JSON形式で出力")
@click.pass_context
def search(ctx, query, results, lang, json_output):
    """意味検索を実行する。

    QUERY: 検索クエリ（日本語・英語どちらも可）
    """
    cfg = ctx.obj["cfg"]
    store = _init_store(cfg)

    if store.count() == 0:
        console.print("[yellow]Corpus is empty. Run 'ingest' first.[/yellow]")
        return

    with console.status(f"[cyan]Searching: {query}[/cyan]"):
        hits = store.search(query=query, n_results=results, language=lang)

    if json_output:
        print(json.dumps(hits, ensure_ascii=False, indent=2))
        return

    console.print(
        f"\n[bold]Search: [cyan]{query}[/cyan][/bold] — {len(hits)} results\n"
    )

    for i, hit in enumerate(hits, 1):
        score_color = (
            "green" if hit["score"] > 0.7 else "yellow" if hit["score"] > 0.5 else "red"
        )
        title = hit["metadata"].get("doc_title", "Unknown")
        heading = hit["metadata"].get("heading_context", "")
        lang_tag = hit["metadata"].get("language", "")

        header = f"[{score_color}]{hit['score']:.3f}[/{score_color}] [{i}] {title}"
        if heading:
            header += f" › [dim]{heading}[/dim]"
        if lang_tag:
            header += f" [{lang_tag}]"

        # anchor_text（エントリー本体）を表示。なければtextにフォールバック
        display = hit["metadata"].get("anchor_text") or hit["text"]
        preview = display[:300] + "..." if len(display) > 300 else display
        console.print(Panel(preview, title=header, border_style="dim"))


# ------------------------------------------------------------------
# analyze
# ------------------------------------------------------------------


@cli.command()
@click.option("--text", default=None, help="分析するテキスト")
@click.option("--file", default=None, help="分析するテキストファイルのパス")
@click.option("--chunk-id", default="", help="チャンクID（省略可）")
@click.option("--json-output", is_flag=True, help="JSON形式で出力")
@click.pass_context
def analyze(ctx, text, file, chunk_id, json_output):
    """テキストの意味論的文脈を抽出・仮説化する。

    坂根構造コーパスの核心機能：パラダイム分類と意味論的仮説生成。
    """
    cfg = ctx.obj["cfg"]

    if file:
        text = Path(file).read_text(encoding="utf-8")
    elif not text:
        console.print("[red]--text または --file を指定してください[/red]")
        sys.exit(1)

    extractor = ContextExtractor(cfg)

    with console.status("[cyan]Analyzing with LLM...[/cyan]"):
        analysis = extractor.extract(text, chunk_id=chunk_id)

    if json_output:
        print(
            json.dumps(
                {
                    "paradigm": analysis.paradigm,
                    "context_summary": analysis.context_summary,
                    "key_concepts": analysis.key_concepts,
                    "implicit_assumptions": analysis.implicit_assumptions,
                    "semantic_hypothesis": analysis.semantic_hypothesis,
                    "relations": analysis.relations,
                    "temporal_marker": analysis.temporal_marker,
                    "provider_used": analysis.provider_used,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    provider_color = "green" if analysis.provider_used == "ollama" else "yellow"
    console.print(
        f"\n[dim]Provider: [{provider_color}]{analysis.provider_used}[/{provider_color}][/dim]\n"
    )

    console.print(
        Panel(
            f"[bold]{analysis.paradigm}[/bold]",
            title="[cyan]Paradigm[/cyan]",
            border_style="cyan",
        )
    )

    console.print(
        Panel(
            analysis.context_summary,
            title="[green]Context Summary[/green]",
            border_style="green",
        )
    )

    console.print(
        Panel(
            analysis.semantic_hypothesis,
            title="[magenta]Semantic Hypothesis[/magenta]",
            border_style="magenta",
        )
    )

    if analysis.key_concepts:
        concepts_text = " • ".join(analysis.key_concepts)
        console.print(Panel(concepts_text, title="Key Concepts", border_style="dim"))

    if analysis.implicit_assumptions:
        for assumption in analysis.implicit_assumptions:
            console.print(f"  [dim]→[/dim] {assumption}")

    if analysis.relations:
        table = Table(title="Concept Relations")
        table.add_column("From", style="cyan")
        table.add_column("Relation", style="yellow")
        table.add_column("To", style="green")
        for rel in analysis.relations:
            table.add_row(
                str(rel.get("from", "")),
                str(rel.get("relation", "")),
                str(rel.get("to", "")),
            )
        console.print(table)


# ------------------------------------------------------------------
# graph-build
# ------------------------------------------------------------------


@cli.command("graph-build")
@click.option(
    "--output", default="data/exports/knowledge_graph.json", help="JSON出力パス"
)
@click.option(
    "--output-graphml",
    default="data/exports/knowledge_graph.graphml",
    help="GraphML出力パス",
)
@click.option("--resume", is_flag=True, help="中断済みの進捗から再開")
@click.option("--batch-size", default=10, help="進捗保存間隔（件数）")
@click.pass_context
def graph_build(ctx, output, output_graphml, resume, batch_size):
    """619件全チャンクをqwen3で分析し知識グラフを構築する。

    初回実行（数時間かかる）:
        python -m cli.corpus_cli graph-build

    クラッシュ後の再開:
        python -m cli.corpus_cli graph-build --resume
    """
    import json as _json
    from pathlib import Path as _Path
    from src.graph.knowledge_graph import KnowledgeGraph
    from src.analyzer.context_extractor import ContextExtractor

    cfg = ctx.obj["cfg"]
    store = _init_store(cfg)
    extractor = ContextExtractor(cfg)

    progress_path = _Path("data/exports/graph_progress.json")
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    # 全チャンク取得
    with console.status("[cyan]Loading all chunks from ChromaDB...[/cyan]"):
        all_chunks = store.get_all_chunks()

    if not all_chunks:
        console.print("[red]Error: No chunks found. Run 'ingest' first.[/red]")
        return

    console.print(f"[green]✓ Loaded {len(all_chunks)} chunks[/green]")

    # resume: 処理済みchunk_idを読み込む
    completed_ids: set[str] = set()
    analyses = []

    if resume and progress_path.exists():
        with open(progress_path, encoding="utf-8") as f:
            progress_data = _json.load(f)
        completed_ids = set(progress_data.get("completed_ids", []))
        analyses = progress_data.get("analyses", [])
        console.print(
            f"[yellow]Resuming: {len(completed_ids)} chunks already processed[/yellow]"
        )

    # 未処理チャンクを抽出
    pending = [c for c in all_chunks if c["chunk_id"] not in completed_ids]
    console.print(f"[cyan]Pending: {len(pending)} chunks to analyze[/cyan]")

    if not pending:
        console.print("[green]All chunks already processed. Building graph...[/green]")
    else:
        # LLM分析ループ
        with console.status("") as status:
            for i, chunk in enumerate(pending, 1):
                chunk_id = chunk["chunk_id"]
                anchor_text = chunk["metadata"].get("anchor_text") or chunk["text"]

                status.update(
                    f"[cyan]Analyzing [{i}/{len(pending)}] {anchor_text[:40]}...[/cyan]"
                )

                analysis = extractor.extract(anchor_text, chunk_id=chunk_id)

                # ContextAnalysisをdict化して保存
                analyses.append(
                    {
                        "chunk_id": analysis.chunk_id,
                        "chunk_text": analysis.chunk_text,
                        "paradigm": analysis.paradigm,
                        "context_summary": analysis.context_summary,
                        "key_concepts": analysis.key_concepts,
                        "implicit_assumptions": analysis.implicit_assumptions,
                        "semantic_hypothesis": analysis.semantic_hypothesis,
                        "relations": analysis.relations,
                        "temporal_marker": analysis.temporal_marker,
                        "provider_used": analysis.provider_used,
                    }
                )
                completed_ids.add(chunk_id)

                # batch_size件ごとに進捗保存
                if i % batch_size == 0:
                    with open(progress_path, "w", encoding="utf-8") as f:
                        _json.dump(
                            {
                                "completed_ids": list(completed_ids),
                                "analyses": analyses,
                            },
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                    console.print(f"[dim]  Progress saved: {i}/{len(pending)}[/dim]")

        # 最終進捗保存
        with open(progress_path, "w", encoding="utf-8") as f:
            _json.dump(
                {
                    "completed_ids": list(completed_ids),
                    "analyses": analyses,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        console.print(f"[green]✓ Analysis complete: {len(analyses)} chunks[/green]")

    # ContextAnalysisオブジェクトに復元してグラフ構築
    from src.analyzer.context_extractor import ContextAnalysis

    analysis_objects = [
        ContextAnalysis(
            chunk_id=a["chunk_id"],
            chunk_text=a["chunk_text"],
            paradigm=a["paradigm"],
            context_summary=a["context_summary"],
            key_concepts=a["key_concepts"],
            implicit_assumptions=a["implicit_assumptions"],
            semantic_hypothesis=a["semantic_hypothesis"],
            relations=a["relations"],
            temporal_marker=a["temporal_marker"],
            provider_used=a["provider_used"],
        )
        for a in analyses
    ]

    with console.status("[cyan]Building knowledge graph...[/cyan]"):
        kg = KnowledgeGraph(min_edge_weight=2)
        kg.build_from_analyses(analysis_objects)

    # エクスポート
    with console.status("[cyan]Exporting...[/cyan]"):
        kg.export(output, format="json")
        kg.export(output_graphml, format="graphml")

    # サマリー
    top = kg.get_top_concepts(n=10)
    table = Table(title="Knowledge Graph Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Total chunks analyzed", str(len(analyses)))
    table.add_row("Graph nodes", str(kg.G.number_of_nodes()))
    table.add_row("Graph edges", str(kg.G.number_of_edges()))
    table.add_row("JSON output", output)
    table.add_row("GraphML output", output_graphml)
    console.print(table)

    console.print("\n[bold]Top 10 concepts:[/bold]")
    for c in top:
        console.print(
            f"  [cyan]{c['concept']}[/cyan] freq={c['frequency']} degree={c['degree']}"
        )


# ------------------------------------------------------------------
# stats
# ------------------------------------------------------------------


@cli.command()
@click.pass_context
def stats(ctx):
    """コーパスの統計情報を表示する。"""
    cfg = ctx.obj["cfg"]
    store = _init_store(cfg)

    sources = store.list_sources()
    total = store.count()

    table = Table(title="[bold]sakane-memex Corpus Statistics[/bold]")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Total chunks", str(total))
    table.add_row("Total source files", str(len(sources)))
    console.print(table)

    if sources:
        console.print("\n[dim]Ingested sources:[/dim]")
        for s in sources:
            console.print(f"  [dim]•[/dim] {s}")


if __name__ == "__main__":
    cli()
