"""Command handlers for the Zypherus REPL."""

from __future__ import annotations

import json
import os
from urllib.parse import urlparse
from typing import Any, Callable, Dict, Tuple


def _is_url(value: str) -> bool:
	try:
		parsed = urlparse(value)
		return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
	except Exception:
		return False


def _is_youtube_url(value: str) -> bool:
	value_lower = value.lower()
	return "youtube.com" in value_lower or "youtu.be" in value_lower


def _parse_ask_options(payload: str) -> Tuple[Dict[str, Any], str]:
	"""Parse lightweight ask options from the start of the payload."""
	options: Dict[str, Any] = {
		"mode": None,
		"style": None,
		"recent_only": None,
	}

	parts = (payload or "").strip().split()
	if not parts:
		return options, ""

	idx = 0
	while idx < len(parts) and parts[idx].startswith("--"):
		flag = parts[idx].lower()
		if flag in {"--fast", "--balanced", "--deep"}:
			options["mode"] = flag.replace("--", "")
		elif flag == "--authoritative":
			options["style"] = "authoritative"
		elif flag == "--practical":
			options["style"] = "practical"
		elif flag == "--recent" or flag == "--recent-only":
			options["recent_only"] = True
		else:
			break
		idx += 1

	remaining = " ".join(parts[idx:])
	return options, remaining


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
			print("[ZYPHERUS] Ingestion complete.")
		except Exception as e:
			print(f"[ZYPHERUS] Ingestion failed: {e}")
		return True, {}

	if cmd.startswith("ingest "):
		parts = cmd.split(None, 2)
		if len(parts) < 2:
			print("Usage: ingest <path> [source]")
			return True, {}
		path = parts[1]
		source = parts[2] if len(parts) >= 3 else os.path.basename(path)
		try:
			if _is_url(path):
				if _is_youtube_url(path):
					from Zypherus.tools.youtube_fetch import ingest_from_youtube_url

					print(f"[ZYPHERUS] Detected YouTube URL: {path}")
					report = ingest_from_youtube_url(ace, path, channel_id=source if source else None)
					print(f"[ZYPHERUS] Status: {report.get('status', 'unknown')}")
					if report.get("status") == "success":
						print(f"[ZYPHERUS] Chunks ingested: {report.get('chunks_ingested', 0)} / {report.get('total_chunks', 0)}")
						print(f"[ZYPHERUS] Ingestion rate: {report.get('ingestion_rate', 'n/a')}")
					else:
						print(f"[ZYPHERUS] Reason: {report.get('reason', 'Unknown')}")
				else:
					if hasattr(ace, "ingest_url"):
						ace.ingest_url(path, source=source)
						print("[ZYPHERUS] Ingestion complete.")
					else:
						print("[ZYPHERUS] ingest_url not available.")
			else:
				if hasattr(ace, "ingest_file"):
					ace.ingest_file(path, source=source)
				else:
					with open(path, "r", encoding="utf-8") as f:
						txt = f.read()
					ace.ingest_document(source, txt, tables=None)
			print("[ZYPHERUS] Ingestion complete.")
		except Exception as e:
			import traceback
			print(f"[ZYPHERUS] Ingestion failed: {type(e).__name__}: {e}")
			print(f"[ZYPHERUS] Traceback:")
			traceback.print_exc()
		return True, {}

	if cmd.startswith("ingest_text"):
		parts = cmd.split(None, 1)
		source = parts[1].strip() if len(parts) >= 2 else None
		
		# Prompt for source if not provided
		if not source:
			print("[ZYPHERUS] Enter source name (e.g., 'python_docs', 'chemistry_facts'):")
			try:
				source = input_line().strip()
			except Exception:
				print("[ZYPHERUS] ❌ Ingestion cancelled.")
				return True, {}
		
		if not source:
			print("[ZYPHERUS] ❌ Source name required. Ingestion cancelled.")
			return True, {}
		
		# Validate source name
		if source == "user_input" or source.startswith("user_input#"):
			print("[ZYPHERUS] ❌ ERROR: Cannot use 'user_input' as a source name.")
			print("[ZYPHERUS] Choose a descriptive name like: 'python_docs', 'chemistry_facts', 'my_notes'")
			return True, {}
		
		print("\n[ZYPHERUS] ════════════════════════════════════════")
		print(f"[ZYPHERUS] INGESTION MODE (source: {source})")
		print("[ZYPHERUS] Paste your content below.")
		print("[ZYPHERUS] End with a blank line (press Enter twice).")
		print("[ZYPHERUS] ════════════════════════════════════════\n")
		
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
		
		# Validation
		if not txt.strip():
			print("[ZYPHERUS] ❌ No text provided. Ingestion cancelled.")
			return True, {}
		
		# Show preview
		preview = txt[:200] + "..." if len(txt) > 200 else txt
		print(f"\n[ZYPHERUS] Preview: {preview}")
		print(f"[ZYPHERUS] Total length: {len(txt)} chars, {len(txt.split())} words")
		print("[ZYPHERUS] Ingesting...")
		
		ace.ingest_document(source, txt, tables=None)
		print("[ZYPHERUS] ✅ Ingestion complete.")
		return True, {}

	if cmd.startswith("ingest_url "):
		parts = cmd.split(None, 2)
		if len(parts) < 2:
			print("Usage: ingest_url <url> [source]")
			return True, {}
		url = parts[1]
		source = parts[2].strip() if len(parts) >= 3 else None
		if source == "user_input" or (source and source.startswith("user_input#")):
			print("[ZYPHERUS] ❌ ERROR: Cannot use 'user_input' as a source name.")
			return True, {}
		try:
			if _is_youtube_url(url):
				from Zypherus.tools.youtube_fetch import ingest_from_youtube_url

				print(f"[ZYPHERUS] Detected YouTube URL: {url}")
				report = ingest_from_youtube_url(ace, url, channel_id=source if source else None)
				print(f"[ZYPHERUS] Status: {report.get('status', 'unknown')}")
				if report.get("status") == "success":
					print(f"[ZYPHERUS] Chunks ingested: {report.get('chunks_ingested', 0)} / {report.get('total_chunks', 0)}")
					print(f"[ZYPHERUS] Ingestion rate: {report.get('ingestion_rate', 'n/a')}")
				else:
					print(f"[ZYPHERUS] Reason: {report.get('reason', 'Unknown')}")
			elif hasattr(ace, "ingest_url"):
				ace.ingest_url(url, source=source)
				print("[ZYPHERUS] Ingestion complete.")
			else:
				print("[ZYPHERUS] ingest_url not available.")
		except Exception as e:
			import traceback
			print(f"[ZYPHERUS] Ingestion failed: {type(e).__name__}: {e}")
			print(f"[ZYPHERUS] Traceback:")
			traceback.print_exc()
		return True, {}

	if cmd.startswith("dump") or cmd == "memory":
		ace.dump_memory()
		return True, {}

	if cmd.strip() == "rebuild_beliefs":
		ace.rebuild_beliefs_from_memory()
		print("[ZYPHERUS] Belief rebuild complete.")
		return True, {}
	
	if cmd.strip() == "validate_memory":
		try:
			from Zypherus.utils.memory_validator import MemoryValidator
			result = MemoryValidator.validate_memory_file("data/memory/memory.json")
			print(f"\n[ZYPHERUS] Memory Validation:")
			print(f"  Total entries: {result['total']}")
			print(f"  Valid entries: {result['valid']}")
			print(f"  Invalid entries: {result['invalid']}")
			if result['invalid'] > 0:
				print(f"\n[ZYPHERUS] Invalid entries found:")
				for e in result['invalid_entries'][:5]:  # Show first 5
					print(f"  - Source: {e['source']}")
					print(f"    Reason: {e['reason']}")
					print(f"    Preview: {e['text_preview']}...")
				if len(result['invalid_entries']) > 5:
					print(f"  ... and {len(result['invalid_entries']) - 5} more")
				print(f"\n[ZYPHERUS] Run 'clean_memory' to remove invalid entries.")
		except Exception as e:
			print(f"[ZYPHERUS] Validation failed: {e}")
		return True, {}
	
	if cmd.strip() == "clean_memory":
		try:
			from Zypherus.utils.memory_validator import MemoryValidator
			print("[ZYPHERUS] Cleaning memory (creating backup)...")
			result = MemoryValidator.clean_memory_file(backup=True)
			print(f"[ZYPHERUS] ✅ Memory cleaned:")
			print(f"  Removed: {result['removed']} entries")
			print(f"  Kept: {result['kept']} entries")
			print(f"  Backup: {result['backup_path']}")
			print(f"\n[ZYPHERUS] Restart Zypherus to reload cleaned memory.")
		except Exception as e:
			print(f"[ZYPHERUS] Clean failed: {e}")
		return True, {}

	if cmd.startswith("ask "):
		payload = cmd.replace("ask ", "", 1).strip()
		debug = False
		if payload.startswith("--debug "):
			debug = True
			payload = payload.replace("--debug ", "", 1).strip()

		options, payload = _parse_ask_options(payload)

		style_hint = None
		if options.get("style") == "authoritative":
			style_hint = "Prefer primary sources, be precise, avoid speculation, and keep tone formal."
		elif options.get("style") == "practical":
			style_hint = "Focus on actionable guidance and concrete steps when possible."

		res = ace.answer(payload, style_hint=style_hint)
		if isinstance(res, dict):
			# Auto-fallback to web search when memory is insufficient.
			if res.get("reason") in {"no_memory", "insufficient_coverage", "low_confidence"} and hasattr(ace, "answer_with_web"):
				res = ace.answer_with_web(
					payload,
					mode=options.get("mode"),
					style=options.get("style"),
					recent_only=options.get("recent_only"),
				)

			if res.get("clarification"):
				print("ACE (clarify):", res["clarification"])
			else:
				print("\nACE Answer:\n", res.get("answer"))
				if debug:
					import json
					print("\nSources:", res.get("sources"))
					print("\nReasoning:", json.dumps(res.get("reasoning"), indent=2))
					print("\nVerification:", json.dumps(res.get("verification"), indent=2))
		else:
			print("\nACE:", res)
		return True, {}

	if cmd.startswith("ask_web "):
		payload = cmd.replace("ask_web ", "", 1).strip()
		debug = False
		if payload.startswith("--debug "):
			debug = True
			payload = payload.replace("--debug ", "", 1).strip()

		options, payload = _parse_ask_options(payload)

		if not payload:
			print("Usage: ask_web <question>")
			return True, {}

		if not hasattr(ace, "answer_with_web"):
			print("[ZYPHERUS] ask_web not available.")
			return True, {}

		res = ace.answer_with_web(
			payload,
			mode=options.get("mode"),
			style=options.get("style"),
			recent_only=options.get("recent_only"),
		)
		if isinstance(res, dict):
			if res.get("clarification"):
				print("ACE (clarify):", res["clarification"])
			else:
				print("\nACE Answer:\n", res.get("answer"))
				if debug:
					import json
					print("\nWeb Sources:", res.get("web_sources"))
					print("\nSources:", res.get("sources"))
					print("\nSelf Critique:", json.dumps(res.get("self_critique"), indent=2))
					print("\nDisagreements:", json.dumps(res.get("disagreements"), indent=2))

					print("\nReasoning:", json.dumps(res.get("reasoning"), indent=2))
					print("\nVerification:", json.dumps(res.get("verification"), indent=2))
		else:
			print("\nACE:", res)
		return True, {}

	# YouTube commands
	if cmd == "youtube_channels":
		try:
			from Zypherus.tools.youtube_curator import YouTubeCurator
			curator = YouTubeCurator()
			curator.add_recommended_channels()
			channels = curator.list_channels()
			print("\n[YOUTUBE] Trusted Channels:")
			print("=" * 70)
			for ch in channels:
				print(f"\n{ch.channel_name} (@{ch.owner})")
				print(f"  ID: {ch.channel_id}")
				print(f"  Trust: {ch.trust_level.value}")
				print(f"  Topics: {', '.join(ch.expertise[:3])}")
				print(f"  Max videos/month: {ch.max_videos_per_month}")
			print()
		except Exception as e:
			print(f"[YOUTUBE] Error: {e}")
		return True, {}

	if cmd.startswith("youtube_preview "):
		parts = cmd.split(None, 3)
		if len(parts) < 4:
			print("Usage: youtube_preview <transcript_file> <channel_id> <video_title>")
			return True, {}
		transcript_path = parts[1]
		channel_id = parts[2]
		video_title = parts[3]
		try:
			from Zypherus.tools.youtube_zypherus import YouTubeIngestionManager
			with open(transcript_path, 'r', encoding='utf-8') as f:
				transcript = f.read()
			manager = YouTubeIngestionManager(ace)
			preview = manager.dry_run_review(transcript, "preview", channel_id, video_title)
			print(f"\n[YOUTUBE] Preview: {video_title}")
			print(f"  Channel: {preview.get('channel', 'N/A')}")
			print(f"  Total segments: {preview.get('total_segments', 0)}")
			print(f"  Chunks to ingest: {preview.get('total_chunks', 0)}")
			if 'sample_chunks' in preview:
				print(f"\n  Sample chunks:")
				for chunk in preview['sample_chunks'][:5]:
					print(f"    - {chunk['topic']}: {chunk['content'][:60]}...")
		except Exception as e:
			print(f"[YOUTUBE] Preview failed: {e}")
		return True, {}

	if cmd.startswith("youtube_ingest "):
		parts = cmd.split(None, 4)
		if len(parts) < 2:
			print("Usage: youtube_ingest <transcript_file_or_url> [channel_id] [video_id] [video_title]")
			return True, {}
		
		first_arg = parts[1]
		
		# Check if first argument is a YouTube URL
		if 'youtube.com' in first_arg or 'youtu.be' in first_arg:
			# URL-based ingestion (auto-fetch transcript)
			try:
				from Zypherus.tools.youtube_fetch import ingest_from_youtube_url
				channel_id = parts[2] if len(parts) > 2 else None
				
				print(f"\n[YOUTUBE] Fetching transcript from: {first_arg}")
				print("=" * 70)
				report = ingest_from_youtube_url(ace, first_arg, channel_id)
				
				print(f"Status: {report['status']}")
				if report['status'] == 'success':
					print(f"Chunks ingested: {report['chunks_ingested']} / {report['total_chunks']}")
					print(f"Ingestion rate: {report['ingestion_rate']}")
				else:
					print(f"Reason: {report.get('reason', 'Unknown')}")
			except Exception as e:
				print(f"[YOUTUBE] Ingestion failed: {e}")
			return True, {}
		else:
			# File-based ingestion (read from local file)
			if len(parts) < 5:
				print("Usage: youtube_ingest <transcript_file> <channel_id> <video_id> <video_title>")
				return True, {}
			transcript_path = parts[1]
			channel_id = parts[2]
			video_id = parts[3]
			video_title = parts[4]
			try:
				from Zypherus.tools.youtube_zypherus import YouTubeIngestionManager
				with open(transcript_path, 'r', encoding='utf-8') as f:
					transcript = f.read()
				manager = YouTubeIngestionManager(ace)
				print(f"\n[YOUTUBE] Ingesting: {video_title}")
				print(f"  Channel: {channel_id}")
				print(f"  Video ID: {video_id}")
				report = manager.ingest_from_transcript(
					transcript, video_id, channel_id, video_title, f"https://youtube.com/watch?v={video_id}"
				)
				print(f"\n[YOUTUBE] Status: {report['status']}")
				if report['status'] == 'success':
					print(f"  Chunks ingested: {report['chunks_ingested']} / {report['total_chunks']}")
					print(f"  Filtered out: {report['filtered_out']}")
					print(f"  Ingestion rate: {report['ingestion_rate']}")
				else:
					print(f"  Reason: {report.get('reason', 'Unknown')}")
			except Exception as e:
				print(f"[YOUTUBE] Ingestion failed: {e}")
			return True, {}

	if cmd == "youtube_stats":
		try:
			from Zypherus.tools.youtube_zypherus import YouTubeIngestionManager
			manager = YouTubeIngestionManager(ace)
			stats = manager.get_ingestion_stats()
			print("\n[YOUTUBE] Ingestion Statistics:")
			print("=" * 70)
			print(f"  Total videos processed: {stats.get('total_videos_processed', 0)}")
			print(f"  Total chunks ingested: {stats.get('total_chunks_ingested', 0)}")
			print(f"  Unique channels: {stats.get('unique_channels', 0)}")
			if stats.get('total_videos_processed', 0) > 0:
				print(f"  Avg chunks per video: {stats.get('avg_chunks_per_video', 0):.1f}")
			print()
		except Exception as e:
			print(f"[YOUTUBE] Error: {e}")
		return True, {}

	if cmd == "youtube_log" or cmd == "youtube_ingested":
		try:
			from pathlib import Path
			import json
			
			# Find ingestion log file
			log_file = Path("data/ingestion/youtube_ingestions.json")
			if not log_file.exists():
				print("[YOUTUBE] No ingestions logged yet.")
				return True, {}
			
			with open(log_file, 'r') as f:
				log_data = json.load(f)
			
			ingestions = log_data.get('ingestions', [])
			if not ingestions:
				print("[YOUTUBE] No ingestions logged yet.")
				return True, {}
			
			print(f"\n[YOUTUBE] Ingestion History ({len(ingestions)} videos)")
			print("=" * 70)
			
			for i, entry in enumerate(ingestions[-10:], 1):  # Show last 10
				print(f"\n{i}. {entry.get('video_title', 'Unknown')}")
				print(f"   Video ID: {entry.get('video_id')}")
				print(f"   Channel: {entry.get('channel')}")
				print(f"   Ingested: {entry.get('chunks_ingested')} / {entry.get('total_chunks')} chunks")
				print(f"   Rate: {entry.get('ingestion_rate')}")
				print(f"   Time: {entry.get('timestamp')}")
			
			print()
		except Exception as e:
			print(f"[YOUTUBE] Error reading ingestion log: {e}")
		return True, {}

	# Combined: Ask question, ingest YouTube video, then re-answer
	if cmd.startswith("ask_then_youtube "):
		parts = cmd.split(None, 5)
		if len(parts) < 6:
			print("Usage: ask_then_youtube <question> <transcript_file> <channel_id> <video_id> <video_title>")
			return True, {}
		question = parts[1]
		transcript_path = parts[2]
		channel_id = parts[3]
		video_id = parts[4]
		video_title = parts[5]
		
		try:
			# Step 1: Ask the initial question (will likely fail/refuse)
			print(f"\n[ASK] Initial question: {question}")
			print("=" * 70)
			initial_result = ace.answer(question)
			if isinstance(initial_result, dict):
				print(f"Initial answer: {initial_result.get('answer')}")
				if initial_result.get('reason'):
					print(f"Reason: {initial_result.get('reason')}")
			else:
				print(f"Initial answer: {initial_result}")
			
			# Step 2: Ingest the YouTube transcript
			print(f"\n[YOUTUBE] Ingesting video: {video_title}")
			print("=" * 70)
			from Zypherus.tools.youtube_zypherus import YouTubeIngestionManager
			with open(transcript_path, 'r', encoding='utf-8') as f:
				transcript = f.read()
			manager = YouTubeIngestionManager(ace)
			report = manager.ingest_from_transcript(
				transcript, video_id, channel_id, video_title, f"https://youtube.com/watch?v={video_id}"
			)
			print(f"Status: {report['status']}")
			if report['status'] == 'success':
				print(f"Chunks ingested: {report['chunks_ingested']} / {report['total_chunks']}")
				print(f"Ingestion rate: {report['ingestion_rate']}")
			else:
				print(f"Reason: {report.get('reason', 'Unknown')}")
			
			# Step 3: Re-ask the question with new knowledge
			print(f"\n[RE-ASK] Question with knowledge from video:")
			print("=" * 70)
			new_result = ace.answer(question)
			if isinstance(new_result, dict):
				print(f"Updated answer: {new_result.get('answer')}")
				print(f"Confidence: {new_result.get('confidence', 'N/A')}")
				sources = new_result.get('sources', [])
				if sources:
					print(f"Sources: {', '.join(sources)}")
			else:
				print(f"Updated answer: {new_result}")
				
		except FileNotFoundError:
			print(f"[ERROR] Transcript file not found: {transcript_path}")
		except Exception as e:
			print(f"[ERROR] Failed: {e}")
		return True, {}

	print("Unknown command.")
	return True, {}

