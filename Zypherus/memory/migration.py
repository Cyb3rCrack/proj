"""
Integration layer connecting 4-layer storage to existing Zypherus memory systems.

Maps existing claims, definitions, and ontology to the appropriate layers.
"""

from __future__ import annotations

import json
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from logging import getLogger

from .storage_backend import (
    RawCapture,
    StructuredPage,
    Section,
    ChunkedContent,
    LearnedFact,
    StorageLayerOrchestrator,
)

logger = getLogger(__name__)


class ExistingMemoryAdapter:
    """
    Adapt existing JSON memory files to 4-layer architecture.
    
    Maps:
    - claims.json → Learned Knowledge Layer (Layer 4)
    - definitions.json → Semantic Layer (Layer 3)
    - ontology.json → Semantic Layer (Layer 3)
    - memory.json → Existing embeddings → Semantic Layer (Layer 3)
    """
    
    def __init__(self, orchestrator: StorageLayerOrchestrator, memory_paths: Dict[str, Path]):
        """
        Args:
            orchestrator: StorageLayerOrchestrator instance
            memory_paths: Dict with keys: claims, definitions, ontology, memory
        """
        self.orchestrator = orchestrator
        self.memory_paths = memory_paths
        self.migration_log: List[Dict[str, Any]] = []
    
    def migrate_claims_to_learned_knowledge(self, claims_file: Path) -> Dict[str, Any]:
        """
        Migrate claims.json → Learned Knowledge Layer (Layer 4).
        
        Each claim becomes a LearnedFact with confidence from the claim's confidence score.
        """
        logger.info(f"Migrating claims from {claims_file}")
        
        if not claims_file.exists():
            logger.warning(f"Claims file not found: {claims_file}")
            return {"migrated": 0, "skipped": 0}
        
        with open(claims_file) as f:
            claims_data = json.load(f)
        
        claims = claims_data.get("claims", {})
        migrated = 0
        skipped = 0
        
        for claim_id, claim in claims.items():
            try:
                # Map claim fields to LearnedFact
                fact_id = f"migrated_claim:{claim_id[:16]}"
                
                # Skip if already migrated
                if self.orchestrator.learned.get_fact(fact_id):
                    skipped += 1
                    continue
                
                # Create learned fact from claim
                fact = LearnedFact(
                    fact_id=fact_id,
                    statement=claim.get("raw", ""),
                    sources=[
                        ev for ev in claim.get("supporting_evidence", [])
                    ],
                    confidence=claim.get("confidence", 0.5),
                    tags=[claim.get("domain", "general")],
                    metadata={
                        "original_claim_id": claim_id,
                        "subject": claim.get("subject"),
                        "predicate": claim.get("predicate"),
                        "object": claim.get("object"),
                        "modality": claim.get("modality"),
                        "created": claim.get("created"),
                    },
                )
                
                self.orchestrator.learned.store_fact(fact)
                migrated += 1
                
                self.migration_log.append({
                    "type": "claim_to_learned_fact",
                    "from": claim_id,
                    "to": fact_id,
                    "confidence": fact.confidence,
                })
                
            except Exception as e:
                logger.error(f"Error migrating claim {claim_id}: {e}")
                skipped += 1
        
        result = {
            "layer": "learned_knowledge",
            "migrated": migrated,
            "skipped": skipped,
            "source_file": str(claims_file),
        }
        
        logger.info(
            f"Claims migration complete: {migrated} migrated, {skipped} skipped"
        )
        return result
    
    def migrate_definitions_to_semantic(self, definitions_file: Path) -> Dict[str, Any]:
        """
        Migrate definitions.json → Semantic Layer (Layer 3).
        
        Each definition becomes a high-importance ChunkedContent.
        """
        logger.info(f"Migrating definitions from {definitions_file}")
        
        if not definitions_file.exists():
            logger.warning(f"Definitions file not found: {definitions_file}")
            return {"migrated": 0, "skipped": 0}
        
        with open(definitions_file) as f:
            defs_data = json.load(f)
        
        definitions = defs_data if isinstance(defs_data, dict) else defs_data.get("definitions", {})
        migrated = 0
        skipped = 0
        
        for def_id, definition in definitions.items():
            try:
                # Compute content hash for deduplication
                if isinstance(definition, dict):
                    content = definition.get("text", json.dumps(definition))
                    term = definition.get("term", def_id)
                else:
                    content = str(definition)
                    term = def_id
                
                content_hash = hashlib.md5(content.encode()).hexdigest()
                chunk_id = f"migrated_def:{content_hash[:16]}"
                
                # Skip if already stored
                if self.orchestrator.semantic.get_chunk(chunk_id):
                    skipped += 1
                    continue
                
                # Create chunked content from definition
                chunk = ChunkedContent(
                    chunk_id=chunk_id,
                    source_url="migrated:definitions.json",
                    page_id="definitions",
                    content_type="definition",
                    text=content,
                    topic_tags=[term.lower()],
                    importance_score=0.9,  # Definitions are high-value
                    information_density=0.8,
                )
                
                stored = self.orchestrator.semantic.store_chunk(chunk)
                if stored:
                    migrated += 1
                else:
                    skipped += 1
                
                self.migration_log.append({
                    "type": "definition_to_chunk",
                    "from": def_id,
                    "to": chunk_id,
                    "importance": 0.9,
                })
                
            except Exception as e:
                logger.error(f"Error migrating definition {def_id}: {e}")
                skipped += 1
        
        result = {
            "layer": "semantic",
            "migrated": migrated,
            "skipped": skipped,
            "source_file": str(definitions_file),
        }
        
        logger.info(
            f"Definitions migration complete: {migrated} migrated, {skipped} skipped"
        )
        return result
    
    def migrate_embeddings_to_semantic(self, embeddings_file: Path) -> Dict[str, Any]:
        """
        Migrate existing embeddings from memory.json → Semantic Layer (Layer 3).
        
        Preserves embedding vectors and associates with semantic chunks.
        """
        logger.info(f"Migrating embeddings from {embeddings_file}")
        
        if not embeddings_file.exists():
            logger.warning(f"Embeddings file not found: {embeddings_file}")
            return {"migrated": 0, "skipped": 0}
        
        with open(embeddings_file) as f:
            embeddings_data = json.load(f)
        
        entries = embeddings_data.get("entries", [])
        migrated = 0
        skipped = 0
        
        for entry in entries:
            try:
                chunk_id = entry.get("id", "")
                if not chunk_id:
                    skipped += 1
                    continue
                
                # Skip if already stored
                if self.orchestrator.semantic.get_chunk(chunk_id):
                    skipped += 1
                    continue
                
                # Parse ID: "definition:Water#hash"
                content_type = "definition"
                if "#" in chunk_id:
                    content_type = chunk_id.split(":")[0]
                
                # Create chunk with preserved embedding
                chunk = ChunkedContent(
                    chunk_id=chunk_id,
                    source_url="migrated:memory.json",
                    page_id="legacy",
                    content_type=content_type,
                    text=entry.get("text", chunk_id),
                    embedding=entry.get("embedding"),
                    topic_tags=entry.get("tags", []),
                    importance_score=entry.get("importance", 0.7),
                    information_density=0.7,
                )
                
                stored = self.orchestrator.semantic.store_chunk(chunk)
                if stored:
                    migrated += 1
                else:
                    skipped += 1
                
            except Exception as e:
                logger.error(f"Error migrating embedding {entry.get('id', '?')}: {e}")
                skipped += 1
        
        result = {
            "layer": "semantic",
            "migrated": migrated,
            "skipped": skipped,
            "source_file": str(embeddings_file),
        }
        
        logger.info(
            f"Embeddings migration complete: {migrated} migrated, {skipped} skipped"
        )
        return result
    
    def run_full_migration(self) -> Dict[str, Any]:
        """Run all migrations."""
        logger.info("Starting full migration to 4-layer storage")
        
        results = {
            "timestamp": time.time(),
            "migrations": [],
        }
        
        # Migrate claims → Layer 4
        if "claims" in self.memory_paths:
            results["migrations"].append(
                self.migrate_claims_to_learned_knowledge(self.memory_paths["claims"])
            )
        
        # Migrate definitions → Layer 3
        if "definitions" in self.memory_paths:
            results["migrations"].append(
                self.migrate_definitions_to_semantic(self.memory_paths["definitions"])
            )
        
        # Migrate embeddings → Layer 3
        if "memory" in self.memory_paths:
            results["migrations"].append(
                self.migrate_embeddings_to_semantic(self.memory_paths["memory"])
            )
        
        # Store migration log
        log_file = Path("migration_log.json")
        with open(log_file, "w") as f:
            json.dump({
                "timestamp": results["timestamp"],
                "migrations_summary": results["migrations"],
                "detailed_log": self.migration_log,
            }, f, indent=2)
        
        logger.info(f"Migration log saved to {log_file}")
        return results


class MemoryLayerBridge:
    """
    Bridge between 4-layer storage and ACE's runtime memory systems.
    
    Provides interfaces for:
    - Querying across all layers
    - Writing updates back to JSON files
    - Maintaining backward compatibility
    """
    
    def __init__(self, orchestrator: StorageLayerOrchestrator):
        self.orchestrator = orchestrator
    
    def export_learned_facts_to_claims(self, output_file: Path) -> Dict[str, Any]:
        """
        Export learned facts layer back to claims.json format for compatibility.
        """
        facts = self.orchestrator.learned.get_active_facts()
        
        claims = {}
        for fact in facts:
            claim_id = hashlib.sha256(fact.statement.encode()).hexdigest()
            
            claims[claim_id] = {
                "id": claim_id,
                "raw": fact.statement,
                "confidence": fact.confidence,
                "created": fact.created_at,
                "sources": fact.sources,
                "tags": fact.tags,
                "domain": fact.tags[0] if fact.tags else "general",
                "_from_learned_layer": True,
            }
        
        output = {
            "schema_version": 2,
            "saved_at": time.time(),
            "claims": claims,
            "note": "Exported from 4-layer learned knowledge layer",
        }
        
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Exported {len(claims)} facts to {output_file}")
        return {"exported_facts": len(claims)}
    
    def export_semantic_chunks_to_definitions(self, output_file: Path) -> Dict[str, Any]:
        """
        Export high-value semantic chunks back to definitions.json format.
        """
        chunks = [
            c for c in self.orchestrator.semantic.chunks.values()
            if c.content_type == "definition"
        ]
        
        definitions = {}
        for chunk in chunks:
            def_id = hashlib.md5(chunk.text.encode()).hexdigest()
            
            definitions[def_id] = {
                "term": chunk.topic_tags[0] if chunk.topic_tags else "unnamed",
                "text": chunk.text,
                "source": chunk.source_url,
                "importance": chunk.importance_score,
            }
        
        with open(output_file, "w") as f:
            json.dump(definitions, f, indent=2)
        
        logger.info(f"Exported {len(definitions)} definitions to {output_file}")
        return {"exported_definitions": len(definitions)}
    
    def search_across_layers(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across all 4 layers for relevant content.
        """
        results = {
            "semantic_chunks": [],
            "learned_facts": [],
            "count": 0,
        }
        
        query_lower = query.lower()
        
        # Search semantic layer
        for chunk in self.orchestrator.semantic.chunks.values():
            if query_lower in chunk.text.lower() or any(
                query_lower in tag for tag in chunk.topic_tags
            ):
                results["semantic_chunks"].append({
                    "type": "semantic_chunk",
                    "id": chunk.chunk_id,
                    "content_type": chunk.content_type,
                    "text": chunk.text[:200],
                    "importance": chunk.importance_score,
                })
        
        # Search learned facts
        for fact in self.orchestrator.learned.get_active_facts():
            if query_lower in fact.statement.lower():
                results["learned_facts"].append({
                    "type": "learned_fact",
                    "id": fact.fact_id,
                    "statement": fact.statement[:200],
                    "confidence": fact.confidence,
                })
        
        results["count"] = len(results["semantic_chunks"]) + len(results["learned_facts"])
        return results


def create_migration_example():
    """Example: Migrating existing memory to 4-layer storage."""
    storage_path = Path("./storage")
    orchestrator = StorageLayerOrchestrator(storage_path)
    
    # Define paths to existing memory files
    memory_paths = {
        "claims": Path("claims.json"),
        "definitions": Path("definitions.json"),
        "memory": Path("memory.json"),
    }
    
    # Run migration
    adapter = ExistingMemoryAdapter(orchestrator, memory_paths)
    results = adapter.run_full_migration()
    
    print(f"Migration results: {json.dumps(results, indent=2)}")
    
    # Export back to JSON for compatibility
    bridge = MemoryLayerBridge(orchestrator)
    bridge.export_learned_facts_to_claims(Path("claims_v2.json"))
    bridge.export_semantic_chunks_to_definitions(Path("definitions_v2.json"))
    
    # Print storage stats
    orchestrator.print_stats()
