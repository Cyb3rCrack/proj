"""Entity indexing and normalization utilities."""

import re
import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


def normalize_entity(entity: str) -> str:
    """Normalize entity name for indexing.
    
    Args:
        entity: Raw entity name
    
    Returns:
        Normalized entity name (lowercase, alphanumeric + spaces)
    """
    normalized = re.sub(r"[^a-z0-9 ]+", "", entity.lower()).strip()
    return normalized


def add_entities(entity_index: Dict[str, Set[str]], 
                entry_id: str, 
                entities: List[str]) -> None:
    """Add entities to index with normalization.
    
    Args:
        entity_index: Entity name to entry IDs mapping
        entry_id: ID of entry being indexed
        entities: List of entity names to add
    """
    for entity in entities or []:
        if not entity:
            continue
        
        normalized = normalize_entity(entity)
        if not normalized:
            continue
        
        # Use sets for O(1) membership checking
        if normalized not in entity_index:
            entity_index[normalized] = set()
        
        entity_index[normalized].add(entry_id)
        logger.debug(f"Added entity '{normalized}' for entry {entry_id}")


def remove_entity(entity_index: Dict[str, Set[str]], 
                 entry_id: str,
                 entity: str) -> None:
    """Remove entity reference from index.
    
    Args:
        entity_index: Entity name to entry IDs mapping
        entry_id: ID of entry to remove
        entity: Entity name to remove
    """
    normalized = normalize_entity(entity)
    if not normalized:
        return
    
    if normalized in entity_index:
        entity_index[normalized].discard(entry_id)
        if not entity_index[normalized]:
            del entity_index[normalized]
            logger.debug(f"Removed empty entity index for '{normalized}'")


def search_entities(entity_index: Dict[str, Set[str]], 
                   query: str) -> Dict[str, Set[str]]:
    """Search entities by partial match.
    
    Args:
        entity_index: Entity name to entry IDs mapping
        query: Search query
    
    Returns:
        Matching entities and their entry IDs
    """
    query_normalized = normalize_entity(query)
    if not query_normalized:
        return {}
    
    results = {}
    for entity_name, entry_ids in entity_index.items():
        if query_normalized in entity_name or entity_name in query_normalized:
            results[entity_name] = entry_ids
    
    return results

