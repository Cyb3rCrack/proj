"""Command handlers for the ACE REPL."""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Tuple


InputFn = Callable[[str], str]


def handle_command(ace: Any, cmd: str, *, input_line: Callable[[], str]) -> Tuple[bool, Dict[str, Any]]:
	"""Handle a single REPL command.

	Returns (should_continue, debug_info).
	"""
	cmd = (cmd or "").strip()
	if not cmd:
		return True, {}

	if cmd.lower() in {"exit", "quit"}:
		return False, {}

	if cmd.startswith("ingest_file "):
		parts = cmd.split(None, 2)
		if len(parts) < 3:
			print("Usage: ingest_file <source> <path>")
			return True, {}
		source = parts[1]
		path = parts[2]
		try:
			if hasattr(ace, "ingest_file"):
				ace.ingest_file(path, source=source)
			else:
				with open(path, "r", encoding="utf-8") as f:
					txt = f.read()
				ace.ingest_document(source, txt, tables=None)
			print("[ACE] Ingestion complete.")
		except Exception as e:
			print(f"[ACE] Ingestion failed: {e}")
		return True, {}

	if cmd.startswith("ingest "):
		parts = cmd.split(None, 2)
		if len(parts) < 2:
			print("Usage: ingest <path> [source]")
			return True, {}
		path = parts[1]
		source = parts[2] if len(parts) >= 3 else os.path.basename(path)
		try:
			if hasattr(ace, "ingest_file"):
				ace.ingest_file(path, source=source)
			else:
				with open(path, "r", encoding="utf-8") as f:
					txt = f.read()
				ace.ingest_document(source, txt, tables=None)
			print("[ACE] Ingestion complete.")
		except Exception as e:
			print(f"[ACE] Ingestion failed: {e}")
		return True, {}

	if cmd.startswith("ingest_text "):
		parts = cmd.split(None, 1)
		if len(parts) < 2:
			print("Usage: ingest_text <source>")
			return True, {}
		source = parts[1]
		print("Enter/paste text. End with a blank line.")
		lines = []
		while True:
			try:
				ln = input_line()
			except Exception:
				ln = ""
			if ln == "":
				break
			lines.append(ln)
		txt = "\n".join(lines)
		ace.ingest_document(source, txt, tables=None)
		print("[ACE] Ingestion complete.")
		return True, {}

	if cmd.startswith("dump") or cmd == "memory":
		ace.dump_memory()
		return True, {}

	if cmd.strip() == "rebuild_beliefs":
		ace.rebuild_beliefs_from_memory()
		print("[ACE] Belief rebuild complete.")
		return True, {}

	if cmd.startswith("ask "):
		payload = cmd.replace("ask ", "", 1).strip()
		debug = False
		if payload.startswith("--debug "):
			debug = True
			payload = payload.replace("--debug ", "", 1).strip()

		res = ace.answer(payload)
		if isinstance(res, dict):
			if res.get("clarification"):
				print("ACE (clarify):", res["clarification"])
			else:
				print("\nACE Answer:\n", res.get("answer"))
				print("Confidence:", res.get("confidence"))
				print("Sources:", res.get("sources"))
				if debug:
					print("\nReasoning:", json.dumps(res.get("reasoning"), indent=2))
					print("\nVerification:", json.dumps(res.get("verification"), indent=2))
		else:
			print("\nACE:", res)
		return True, {}

	print("Unknown command.")
	return True, {}

