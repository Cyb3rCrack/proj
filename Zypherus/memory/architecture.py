"""
4-LAYER STORAGE ARCHITECTURE FOR ACE MEMORY SYSTEM
============================================================

Core Principle: Unlimited Ingestion ≠ Unlimited Memory

The system separates concerns across 4 layers:
1. Raw Capture        (Unlimited, Write-Only, Cold)
2. Structural         (Bounded, Reusable)
3. Semantic           (Controlled, Expensive)
4. Learned Knowledge  (Tiny, High-Value)

Each layer has distinct guarantees and eviction policies.


== LAYER 1: RAW CAPTURE ================================================

PURPOSE: Store everything exactly once, cheaply, forever.

PROPERTIES:
  - Write-only (never delete raw)
  - Deduplicated by content hash
  - Compressed storage
  - Never queried directly by AI
  - Archive-style access

WHAT TO STORE:
  - Full HTML
  - Cleaned text
  - Metadata (URL, timestamp, headers, author, version)
  - Content hashes (SHA256)

STORAGE MODEL:
  raw_captures/
  └─ example.com/
     ├─ hash_1.html
     ├─ hash_1.txt
     ├─ hash_1_metadata.json
     └─ index.json

USAGE:
  from Zypherus.memory.storage_backend import RawCaptureStore
  
  store = RawCaptureStore(Path("./storage"))
  
  capture = RawCapture(
      source_url="https://example.com/page",
      raw_html="<html>...</html>",
      cleaned_text="Cleaned text...",
      metadata={"author": "...", "timestamp": "..."}
  )
  
  is_new = store.store_capture(capture)  # True if new, False if duplicate


== LAYER 2: STRUCTURAL ================================================

PURPOSE: Extract and store structured knowledge for rereading.

PROPERTIES:
  - Bounded by page count (not content size)
  - Reusable for multiple purposes
  - Not embeddings yet
  - Enables targeted rereads
  - Supports partial ingestion

WHAT TO EXTRACT:
  - Heading hierarchy
  - Sections with IDs
  - Tables
  - Code blocks
  - Internal links
  - Version markers

STRUCTURE EXAMPLE:
  {
    "url": "...",
    "sections": [
      {
        "id": "intro",
        "title": "Overview",
        "level": 2,
        "content": "...",
        "subsections": ["intro_subsec1"]
      },
      {
        "id": "api",
        "title": "API Reference",
        "level": 2,
        "content": "..."
      }
    ],
    "tables": [...],
    "code_blocks": [...]
  }

USAGE:
  from Zypherus.memory.storage_backend import StructuralLayer, Section
  
  layer = StructuralLayer(Path("./storage"))
  
  sections = [
      Section(
          section_id="intro",
          title="Overview",
          level=2,
          content="This is the content",
          subsections=[]
      )
  ]
  
  page_id = layer.store_page(structured_page)


== LAYER 3: SEMANTIC ================================================

PURPOSE: What the AI can reason over.

PROPERTIES:
  - Controlled budget (importance threshold)
  - High value only
  - Embeddings stored here
  - Supports eviction
  - Queryable by tag/importance

IMPORTANCE SCORING:
  
  Score = 0.5 * type_score
        + 0.2 * density_score
        + 0.2 * novelty_score
        + query_relevance_bonus
  
  Where:
    - type_score: "definition" (0.95) vs "filler" (0.1)
    - density_score: Technical content density
    - novelty_score: 1.0 = completely new, 0.0 = fully duplicated
    - query_relevance_bonus: +0.15 if matches user query
  
  Only store if score >= importance_threshold (default 0.5)

GOOD CANDIDATES FOR SEMANTIC STORAGE:
  ✓ Definitions
  ✓ Explanations
  ✓ Specifications
  ✓ Algorithms
  ✓ Normative statements
  ✓ Rules

BAD CANDIDATES:
  ✗ Navigation
  ✗ Legal boilerplate
  ✗ Repeated examples
  ✗ Table of contents

USAGE:
  from Zypherus.memory.storage_backend import SemanticLayer, ChunkedContent
  
  layer = SemanticLayer(Path("./storage"), importance_threshold=0.5)
  
  chunk = ChunkedContent(
      chunk_id="chunk_xyz",
      source_url="https://example.com",
      page_id="page_abc",
      content_type="definition",
      text="A definition here",
      topic_tags=["concept", "terminology"],
      importance_score=0.85,
      information_density=0.8
  )
  
  stored = layer.store_chunk(chunk)  # Returns True if stored
  
  # Eviction
  layer.evict_chunks(num_to_evict=10)  # Removes lowest-value chunks


== LAYER 4: LEARNED KNOWLEDGE ==========================================

PURPOSE: What the AI actually "knows".

PROPERTIES:
  - Tiny (high signal-to-noise ratio)
  - Synthesized facts
  - Answer-ready statements
  - Confidence + provenance
  - Expiry conditions

WHAT TO STORE:
  - Synthesized facts (not raw text)
  - Inferred rules
  - Validated conclusions
  - Domain-specific knowledge

STRUCTURE:
  {
    "fact_id": "fact:abc123",
    "statement": "Overcommit behavior depends on vm.overcommit_memory",
    "sources": ["chunk_xyz", "chunk_uvw"],
    "confidence": 0.91,
    "created_at": 1738744800,
    "expires_at": 1739349600,  # When to reconsider
    "tags": ["kernel", "memory", "sysadmin"],
    "metadata": {
      "domain": "linux",
      "version": "5.18"
    }
  }

AGING & DECAY:
  - Facts don't delete; they decay
  - Every 30 days: confidence *= 0.9
  - Expired facts marked for reconsideration
  - Keep history of confidence changes

USAGE:
  from ace.memory.storage_backend import LearnedKnowledgeLayer, LearnedFact
  
  layer = LearnedKnowledgeLayer(Path("./storage"))
  
  fact = LearnedFact(
      fact_id="fact:xyz",
      statement="The concept definition",
      sources=["chunk_abc"],
      confidence=0.85,
      expires_at=time.time() + 30*86400,  # 30 days
      tags=["concept", "domain"]
  )
  
  layer.store_fact(fact)
  
  # Get facts (exclude expired)
  active_facts = layer.get_active_facts()
  
  # Decay old facts
  layer.downgrade_old_facts(age_days=30)


== PROGRESSIVE INGESTION PIPELINE ======================================

Flow: Capture → Structure → Score → Embed → Synthesize

STAGE 1: CAPTURE (Layer 1)
  Input: HTML, cleaned text, metadata
  Output: RawCapture (deduplicated by hash)
  Time: Fast (write only)
  Dedup: Content hash
  
  store = RawCaptureStore(...)
  is_new = store.store_capture(capture)

STAGE 2: STRUCTURE (Layer 2)
  Input: RawCapture + parsing results
  Output: StructuredPage (sections, tables, links)
  Time: Moderate (parsing)
  Purpose: Enables rereading, partial ingestion
  
  layer = StructuralLayer(...)
  page_id = layer.store_page(structured_page)

STAGE 3: SCORE IMPORTANCE (Layer 3 prep)
  Input: Text + content_type + context
  Output: Importance score (0.0 - 1.0)
  Time: Fast (heuristics)
  Factors:
    - Content type (definition > explanation > filler)
    - Information density
    - Novelty vs existing
    - Query relevance
  
  from ace.memory.ingestion_pipeline import ImportanceScorer
  
  scorer = ImportanceScorer()
  importance = scorer.score_content_type("definition")
  density = scorer.compute_information_density(text)
  novelty = scorer.compute_novelty(text, existing_texts)

STAGE 4: SEMANTIC STORAGE (Layer 3)
  Input: ChunkedContent + importance_score + embedding
  Output: Stored chunk (if score >= threshold) or rejected
  Time: Expensive (embeddings if included)
  Filter: Only high-importance chunks
  
  layer = SemanticLayer(...)
  stored = layer.store_chunk(chunked_content)
  
  # Eviction if space needed
  layer.evict_chunks(num_to_evict=5)

STAGE 5: SYNTHESIZE (Layer 4)
  Input: High-value semantic chunks
  Output: LearnedFact (synthesized knowledge)
  Time: Very fast (just aggregation)
  Only synthesize: importance_score >= 0.7
  
  from ace.memory.ingestion_pipeline import ProgressiveIngestionPipeline
  
  pipeline = ProgressiveIngestionPipeline(orchestrator)
  fact_id = pipeline.stage_5_synthesize(
      chunk_id="...",
      chunked_content=chunk,
      confidence=0.85,
      expires_days=30
  )


== QUICK START GUIDE ==================================================

1. Initialize orchestrator:
   
   from ace.memory.storage_backend import StorageLayerOrchestrator
   from pathlib import Path
   
   orchestrator = StorageLayerOrchestrator(Path("./memory_storage"))
   orchestrator.print_stats()

2. Full page ingestion:
   
   from ace.memory.ingestion_pipeline import ProgressiveIngestionPipeline
   
   pipeline = ProgressiveIngestionPipeline(orchestrator)
   
   result = pipeline.ingest_page_full_pipeline(
       source_url="https://example.com/page",
       raw_html="<html>...</html>",
       cleaned_text="Cleaned text",
       sections=[...],
       metadata={"author": "..."}
   )
   
   print(f"Stored {result['chunks_stored']} chunks")
   print(f"Synthesized {result['facts_synthesized']} facts")

3. Migrate existing memory:
   
   from ace.memory.migration import ExistingMemoryAdapter
   
   adapter = ExistingMemoryAdapter(
       orchestrator,
       memory_paths={
           "claims": Path("claims.json"),
           "definitions": Path("definitions.json"),
           "memory": Path("memory.json"),
       }
   )
   
   results = adapter.run_full_migration()
   print(f"Migration: {results}")

4. Query across layers:
   
   from ace.memory.migration import MemoryLayerBridge
   
   bridge = MemoryLayerBridge(orchestrator)
   
   results = bridge.search_across_layers("your query")
   print(f"Found {results['count']} results")
   print(f"  Semantic chunks: {len(results['semantic_chunks'])}")
   print(f"  Learned facts: {len(results['learned_facts'])}")

5. Export for compatibility:
   
   bridge.export_learned_facts_to_claims(Path("claims_v2.json"))
   bridge.export_semantic_chunks_to_definitions(Path("definitions_v2.json"))


== EVICTION STRATEGIES =================================================

When to evict:

  SEMANTIC LAYER (Layer 3):
  - When total chunks exceed budget_limit
  - Evict: lowest importance_score first
  - Keep high-density, novel content
  
  LEARNED KNOWLEDGE LAYER (Layer 4):
  - Never delete - only downgrade
  - After expires_at: confidence *= 0.9 per 30 days
  - Mark for reconsideration if confidence drops below 0.5

When NOT to evict:

  RAW CAPTURE LAYER (Layer 1):
  - Never evict (append-only, cheap storage)
  - Use compression
  - Archive to cold storage if needed
  
  STRUCTURAL LAYER (Layer 2):
  - Keep paginated (bounded by URL count)
  - Versions provide history
  - Old versions can be compressed


== VERSIONING & DECAY =================================================

Every stored unit knows:
  - When ingested (created_at)
  - From which version (ingestion_version)
  - When to reconsider (expires_at)
  - Confidence history (metadata)

Example fact lifecycle:
  T=0 days:     confidence = 0.90 (verified)
  T=30 days:    confidence = 0.81 (decayed)
  T=60 days:    confidence = 0.73 (decayed)
  T=90 days:    confidence = 0.65 (marked for review)
  T=120 days:   confidence = 0.59 (needs verification)
  T>180 days:   confidence < 0.30 (very uncertain)


== STORAGE LAYOUT ======================================================

memory_storage/
├─ raw_captures/
│  ├─ index.json
│  ├─ hash_1.json
│  └─ hash_2.json
├─ structural/
│  ├─ index.json
│  ├─ page_id_1.json
│  └─ page_id_2.json
├─ semantic/
│  └─ embeddings.json
└─ learned/
   └─ facts.json


== STATISTICS & MONITORING ============================================

Get stats from all layers:
  
  stats = orchestrator.get_full_stats()
  
  Returns:
  {
    "raw_capture": {
      "total_captures": 1500,
      "total_size_mb": 450,
      "unique_urls": 120
    },
    "structural": {
      "total_pages": 120,
      "total_sections": 450,
      "total_size_mb": 5
    },
    "semantic": {
      "total_chunks": 450,
      "avg_importance_score": 0.72,
      "total_size_mb": 85
    },
    "learned_knowledge": {
      "total_facts": 45,
      "active_facts": 42,
      "expired_facts": 3,
      "avg_confidence": 0.78,
      "size_kb": 12
    },
    "summary": {
      "total_storage_mb": 545
    }
  }

Print formatted stats:
  
  orchestrator.print_stats()


== BEST PRACTICES ======================================================

1. CAPTURE EVERYTHING
   - Raw layer is cheap and append-only
   - Deduplication happens automatically
   - Keep metadata for historical context

2. STRUCTURE SELECTIVELY
   - Extract only meaningful sections
   - Preserve heading hierarchy
   - Link to source raw capture
   - Use version markers

3. SCORE RIGOROUSLY
   - Don't embed everything
   - Use importance scoring
   - Exclude boilerplate, navigation
   - Favor definitions, rules, algorithms

4. SYNTHESIZE CAREFULLY
   - Only high-importance chunks → facts
   - Include full provenance
   - Set reasonable confidence levels
   - Mark expiry conditions

5. MONITOR STORAGE
   - Check layer stats regularly
   - Evict low-value semantic chunks
   - Downgrade aging facts
   - Track migration log

6. QUERY SMART
   - Search across all layers
   - Prioritize learned facts over chunks
   - Check confidence/importance scores
   - Verify provenance


== INTEGRATION WITH ACE MEMORY =========================================

The 4-layer system is designed to coexist with existing Zypherus memory:

- Existing claims.json → Learned Knowledge Layer
- Existing definitions.json → Semantic Layer
- Existing memory.json embeddings → Semantic Layer

Run migration once:
  
  adapter = ExistingMemoryAdapter(orchestrator, memory_paths)
  results = adapter.run_full_migration()

Then maintain dual exports:
  
  bridge = MemoryLayerBridge(orchestrator)
  bridge.export_learned_facts_to_claims(Path("claims_v2.json"))

This provides:
  - Backward compatibility
  - Gradual transition
  - Ability to compare
  - Rollback option
"""

import json
from pathlib import Path


def print_architecture_guide():
    """Print this documentation to console."""
    module_str = __doc__
    print(module_str)


def save_architecture_guide(output_file: Path):
    """Save this documentation to file."""
    with open(output_file, "w") as f:
        f.write(__doc__)
    print(f"Architecture guide saved to {output_file}")


if __name__ == "__main__":
    print_architecture_guide()
