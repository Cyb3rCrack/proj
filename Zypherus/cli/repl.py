"""Interactive REPL for Zypherus."""

from __future__ import annotations

from typing import Any

from .commands import handle_command


def run_repl(ace: Any = None) -> None:
    # Import here to avoid import-order surprises during migration.
    if ace is None:
        import importlib

        ace_mod = importlib.import_module("Zypherus.core.ace")
        ACE = getattr(ace_mod, "ACE")
        ace = ACE()

    print("\nCommands:")
    print("  ingest_file <source> <path>  → ingest local file as document")
    print("  ingest_text [source]         → paste text interactively (source optional)")
    print("  ingest_url <url> [source]    → fetch web page and ingest text")
    print("  ask <question>               → ask Zypherus a question (auto-web if needed)")
    print("    flags: --fast|--balanced|--deep, --authoritative|--practical, --recent")
    print("  ask --debug <question>       → ask and show reasoning/verification")
    print("  ask_web <question>           → search web, ingest top URLs, then answer")
    print("  ask_web --debug <question>   → ask_web with sources/debug details")
    print("    flags: --fast|--balanced|--deep, --authoritative|--practical, --recent")
    print("  ask_then_youtube <q> <f> <ch> <id> <t> → ask, ingest YouTube video, re-answer")
    print("  youtube_channels             → list trusted YouTube channels")
    print("  youtube_preview <f> <ch> <t> → preview YouTube transcript ingestion")
    print("  youtube_ingest <url_or_file> [channel] → ingest YouTube video (URL or transcript file)")
    print("  youtube_log                  → view ingestion history from data/ingestion/youtube_ingestions.json")
    print("  rebuild_beliefs              → rebuild beliefs from stored evidence")
    print("  validate_memory              → check memory for contamination")
    print("  clean_memory                 → remove contaminated entries")
    print("  exit")
    print()
    print("IMPORTANT: Keep ingestion and questions separate!")
    print("  ✅ Correct:   ingest_text python_docs     (then paste knowledge)")
    print("  ❌ Wrong:     ask ingest_text ...         (this contaminates memory)")

    while True:
        try:
            cmd = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        try:
            should_continue, _debug = handle_command(ace, cmd, input_line=lambda: input())
            if not should_continue:
                break
        except Exception as e:
            print(f"[ZYPHERUS] Command failed: {e}")

