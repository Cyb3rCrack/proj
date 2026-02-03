"""Entry point for ACE."""

from __future__ import annotations


def main() -> None:
    from ace.cli.repl import run_repl

    run_repl()


if __name__ == "__main__":
    main()

