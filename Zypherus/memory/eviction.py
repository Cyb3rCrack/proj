"""Memory caps and eviction policies."""

import logging
from typing import Dict, List, Set, Any

logger = logging.getLogger(__name__)


def evict_if_needed(entries: List[Dict[str, Any]], 
                   max_total_entries: int,
                   max_entries_per_source: int,
                   sources: Set[str],
                   entity_index: Dict[str, List[str]]) -> Set[str]:
    """Evict entries if memory limits exceeded.
    
    Args:
        entries: List of memory entries
        max_total_entries: Maximum total entries allowed
        max_entries_per_source: Maximum entries per source
        sources: Set of source dedup keys
        entity_index: Entity to entry ID mapping
    
    Returns:
        Set of removed entry IDs
    """
    if max_total_entries <= 0 and max_entries_per_source <= 0:
        return set()
    
    if not entries:
        return set()

    removed_ids: Set[str] = set()
    removed_dedup: Set[str] = set()

    def _root_source(source: str) -> str:
        """Get root source identifier."""
        return source.split("/")[0] if "/" in source else source

    # Evict by source if limit exceeded
    if max_entries_per_source > 0:
        by_root: Dict[str, List[Dict[str, Any]]] = {}
        for e in entries:
            root = _root_source(e.get("source", ""))
            by_root.setdefault(root, []).append(e)
        
        for _root, lst in by_root.items():
            if len(lst) <= max_entries_per_source:
                continue
            
            lst_sorted = sorted(lst, key=lambda x: float(x.get("ts", 0.0) or 0.0))
            evict_count = len(lst_sorted) - max_entries_per_source
            
            # Prefer evicting evidence entries
            evict = [e for e in lst_sorted if e.get("type") == "evidence"][:evict_count]
            if len(evict) < evict_count:
                extra = [e for e in lst_sorted if e not in evict][:(evict_count - len(evict))]
                evict.extend(extra)
            
            for e in evict:
                removed_ids.add(e.get("id"))
                if e.get("_dedup_key"):
                    removed_dedup.add(e.get("_dedup_key"))

    # Evict oldest if total limit exceeded
    if max_total_entries > 0 and len(entries) - len(removed_ids) > max_total_entries:
        remaining = [e for e in entries if e.get("id") not in removed_ids]
        remaining_sorted = sorted(remaining, key=lambda x: float(x.get("ts", 0.0) or 0.0))
        evict_count = len(remaining_sorted) - max_total_entries
        
        evict = [e for e in remaining_sorted if e.get("type") == "evidence"][:evict_count]
        if len(evict) < evict_count:
            extra = [e for e in remaining_sorted if e not in evict][:(evict_count - len(evict))]
            evict.extend(extra)
        
        for e in evict:
            removed_ids.add(e.get("id"))
            if e.get("_dedup_key"):
                removed_dedup.add(e.get("_dedup_key"))

    # Clean up references
    if removed_ids:
        try:
            for k in removed_dedup:
                if k in sources:
                    sources.remove(k)
        except Exception as e:
            logger.error(f"Failed to clean dedup sources: {e}")
        
        try:
            for ent, ids in list(entity_index.items()):
                if not ids:
                    continue
                new_ids = [i for i in ids if i not in removed_ids]
                if new_ids:
                    entity_index[ent] = new_ids
                else:
                    del entity_index[ent]
        except Exception as e:
            logger.error(f"Failed to clean entity index: {e}")
    
    if removed_ids:
        logger.info(f"Evicted {len(removed_ids)} entries to maintain memory limits")
    
    return removed_ids

