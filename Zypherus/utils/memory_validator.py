"""Memory validation and cleaning utility."""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

logger = logging.getLogger("ZYPHERUS.MemoryValidator")


class MemoryValidator:
    """Validates and cleans memory entries to prevent contamination."""
    
    CONTAMINATED_PHRASES = [
        "i'll be happy to", "please provide", "i'm happy to",
        "what went wrong", "can you give me", "tell me about",
        "i don't have", "i cannot", "as an ai",
        "sorry, i", "i apologize", "thank you for",
        "it seems like there is no", "it seems like you",
        "i'm sorry", "i apologize", "i can't",
    ]
    
    FORBIDDEN_SOURCES = ["user_input", "test", "learning", "debug"]
    
    @staticmethod
    def is_valid_entry(entry: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if a memory entry is valid.
        
        Returns:
            (is_valid, reason)
        """
        text = entry.get("text", "")
        source = entry.get("source", "")
        
        # Check source
        if source == "user_input" or source.startswith("user_input#"):
            return False, "forbidden source: user_input"
        
        # Check for test sources (optional, can be commented out)
        root_source = source.split("#")[0] if "#" in source else source
        if root_source in MemoryValidator.FORBIDDEN_SOURCES:
            return False, f"forbidden source: {root_source}"
        
        # Check text length
        if len(text.split()) < 5:
            return False, "too short (< 5 words)"
        
        # Check for contaminated phrases
        text_lower = text.lower()
        for phrase in MemoryValidator.CONTAMINATED_PHRASES:
            if phrase in text_lower:
                return False, f"contaminated phrase: '{phrase}'"
        
        return True, ""
    
    @staticmethod
    def _resolve_memory_path(path: Optional[str] = None) -> str:
        if path:
            provided = Path(path)
            if provided.exists():
                return str(provided)
            fallback = Path("memory.json")
            if fallback.exists():
                return str(fallback)
            return path
        default_path = Path("data") / "memory" / "memory.json"
        if default_path.exists():
            return str(default_path)
        return "memory.json"

    @staticmethod
    def validate_memory_file(path: Optional[str] = None) -> Dict[str, Any]:
        """Validate a memory.json file and return statistics.
        
        Returns:
            {
                "total": int,
                "valid": int,
                "invalid": int,
                "invalid_entries": List[Dict],
            }
        """
        resolved_path = MemoryValidator._resolve_memory_path(path)
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load memory file: {e}")
            return {"total": 0, "valid": 0, "invalid": 0, "invalid_entries": []}
        
        entries = data.get("entries", [])
        invalid_entries = []
        
        for i, entry in enumerate(entries):
            is_valid, reason = MemoryValidator.is_valid_entry(entry)
            if not is_valid:
                invalid_entries.append({
                    "index": i,
                    "source": entry.get("source"),
                    "text_preview": entry.get("text", "")[:100],
                    "reason": reason,
                })
        
        return {
            "total": len(entries),
            "valid": len(entries) - len(invalid_entries),
            "invalid": len(invalid_entries),
            "invalid_entries": invalid_entries,
        }
    
    @staticmethod
    def clean_memory_file(path: Optional[str] = None, backup: bool = True) -> Dict[str, Any]:
        """Clean a memory.json file by removing invalid entries.
        
        Args:
            path: Path to memory.json
            backup: Whether to create a backup first
            
        Returns:
            {
                "removed": int,
                "kept": int,
                "backup_path": str or None,
            }
        """
        import shutil
        from datetime import datetime
        
        resolved_path = MemoryValidator._resolve_memory_path(path)
        backup_path = None
        if backup:
            backup_path = f"memory_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            shutil.copy(resolved_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
        
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load memory file: {e}")
            return {"removed": 0, "kept": 0, "backup_path": backup_path}
        
        entries = data.get("entries", [])
        valid_entries = []
        
        for entry in entries:
            is_valid, _ = MemoryValidator.is_valid_entry(entry)
            if is_valid:
                valid_entries.append(entry)
        
        removed_count = len(entries) - len(valid_entries)
        
        # Save cleaned memory
        data["entries"] = valid_entries
        with open(resolved_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Cleaned memory: removed {removed_count}, kept {len(valid_entries)}")
        
        return {
            "removed": removed_count,
            "kept": len(valid_entries),
            "backup_path": backup_path,
        }


__all__ = ["MemoryValidator"]
