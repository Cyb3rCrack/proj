"""Interactive REPL for ACE."""

from __future__ import annotations

from typing import Any

from .commands import handle_command


def run_repl(ace: Any = None) -> None:
    # Import here to avoid import-order surprises during migration.
    if ace is None:
        from ace.core.ace import ACE
        ace = ACE()

    print("\nCommands:")
    print("  ingest_file <source> <path>  → ingest local file as document")
    print("  ingest_text <source>         → paste text interactively")
    print("  ask <question>               → ask ACE")
    print("  ask --debug <question>       → ask and show reasoning/verification")
    print("  rebuild_beliefs              → rebuild beliefs from stored evidence")
    print("  exit")

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
            print(f"[ACE] Command failed: {e}")

