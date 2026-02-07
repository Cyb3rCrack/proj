"""Memory cleaning utility to purge polluted entries."""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple


class MemoryCleaner:
	"""Tools for cleaning polluted memory entries."""
	
	@staticmethod
	def scan_memory(memory_path: str = "data/memory/memory.json") -> Dict[str, Any]:
		"""Scan memory for issues and return report."""
		
		with open(memory_path, 'r', encoding='utf-8') as f:
			data = json.load(f)
		
		entries = data.get("entries", [])
		
		report = {
			"total_entries": len(entries),
			"polluted": [],
			"duplicates": [],
			"by_type": {},
			"by_source": {},
		}
		
		# Track duplicates
		seen_texts = {}
		
		for i, entry in enumerate(entries):
			text = entry.get("text", "")
			entry_type = entry.get("type", "evidence")
			source = entry.get("source", "unknown")
			
			# Count by type
			report["by_type"][entry_type] = report["by_type"].get(entry_type, 0) + 1
			
			# Count by source
			report["by_source"][source] = report["by_source"].get(source, 0) + 1
			
			# Check for pollution
			is_polluted, reason = MemoryCleaner._is_polluted(text)
			if is_polluted:
				report["polluted"].append({
					"index": i,
					"id": entry.get("id"),
					"reason": reason,
					"text_preview": text[:100],
					"source": source,
					"type": entry_type,
				})
			
			# Check for duplicates
			text_lower = text.lower().strip()
			if text_lower in seen_texts:
				report["duplicates"].append({
					"index": i,
					"id": entry.get("id"),
					"duplicate_of_index": seen_texts[text_lower],
					"text_preview": text[:100],
				})
			else:
				seen_texts[text_lower] = i
		
		return report
	
	@staticmethod
	def _is_polluted(text: str) -> Tuple[bool, str]:
		"""Check if text is polluted (assistant chatter, errors, etc)."""
		
		pollution_patterns = [
			(r"i'?m happy to help", "Assistant chatter"),
			(r"you didn'?t provide", "Assistant error message"),
			(r"please provide", "Request for input"),
			(r"you can use this as", "Meta instruction"),
			(r"here'?s? (?:how|what|the)", "Conversational prefix"),
			(r"let me (?:help|explain|show)", "Conversational prefix"),
			(r"^i (?:am|can|will)", "First person conversational"),
			(r"error:|exception:|warning:", "Error message"),
			(r"traceback", "Stack trace"),
		]
		
		text_lower = text.lower().strip()
		
		for pattern, reason in pollution_patterns:
			if re.search(pattern, text_lower):
				return True, reason
		
		# Too short
		if len(text.strip()) < 10:
			return True, "Too short"
		
		# Too few words
		if len(text.split()) < 3:
			return True, "Too few words"
		
		return False, ""
	
	@staticmethod
	def clean_memory(
		memory_path: str = "data/memory/memory.json",
		output_path: str = "memory_cleaned.json",
		remove_polluted: bool = True,
		remove_duplicates: bool = True,
		dry_run: bool = False
	) -> Dict[str, Any]:
		"""Clean memory by removing polluted/duplicate entries."""
		
		with open(memory_path, 'r', encoding='utf-8') as f:
			data = json.load(f)
		
		entries = data.get("entries", [])
		original_count = len(entries)
		
		indices_to_remove = set()
		
		# Find polluted entries
		if remove_polluted:
			for i, entry in enumerate(entries):
				text = entry.get("text", "")
				is_polluted, reason = MemoryCleaner._is_polluted(text)
				if is_polluted:
					indices_to_remove.add(i)
		
		# Find duplicates
		if remove_duplicates:
			seen_texts = {}
			for i, entry in enumerate(entries):
				text = entry.get("text", "").lower().strip()
				if text in seen_texts:
					indices_to_remove.add(i)  # Remove the duplicate
				else:
					seen_texts[text] = i
		
		# Remove entries
		if not dry_run and indices_to_remove:
			cleaned_entries = [e for i, e in enumerate(entries) if i not in indices_to_remove]
			data["entries"] = cleaned_entries
			
			# Write cleaned data
			with open(output_path, 'w', encoding='utf-8') as f:
				json.dump(data, f, ensure_ascii=False, indent=2)
		
		return {
			"original_count": original_count,
			"removed_count": len(indices_to_remove),
			"final_count": original_count - len(indices_to_remove),
			"removed_indices": sorted(indices_to_remove),
			"dry_run": dry_run,
			"output_path": output_path if not dry_run else None,
		}
	
	@staticmethod
	def print_report(report: Dict[str, Any]):
		"""Pretty print a scan report."""
		print(f"\n{'='*60}")
		print(f"MEMORY SCAN REPORT")
		print(f"{'='*60}\n")
		
		print(f"Total entries: {report['total_entries']}")
		print(f"Polluted entries: {len(report['polluted'])}")
		print(f"Duplicate entries: {len(report['duplicates'])}")
		
		print(f"\n--- By Type ---")
		for t, count in sorted(report['by_type'].items(), key=lambda x: -x[1]):
			print(f"  {t}: {count}")
		
		print(f"\n--- By Source (top 10) ---")
		for src, count in sorted(report['by_source'].items(), key=lambda x: -x[1])[:10]:
			print(f"  {src}: {count}")
		
		if report['polluted']:
			print(f"\n--- Polluted Entries (first 10) ---")
			for item in report['polluted'][:10]:
				print(f"  [{item['index']}] {item['reason']}: {item['text_preview']}...")
		
		if report['duplicates']:
			print(f"\n--- Duplicate Entries (first 10) ---")
			for item in report['duplicates'][:10]:
				print(f"  [{item['index']}] duplicate of [{item['duplicate_of_index']}]: {item['text_preview']}...")


__all__ = ["MemoryCleaner"]
