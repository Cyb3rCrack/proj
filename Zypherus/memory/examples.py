"""
4-LAYER STORAGE QUICK REFERENCE
================================

Copy-paste ready examples for common tasks.


TASK 1: Initialize Storage System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from pathlib import Path
from Zypherus.memory import StorageLayerOrchestrator

# Create orchestrator
orch = StorageLayerOrchestrator(Path("./storage"))

# Print stats
orch.print_stats()


TASK 2: Store Raw Content
~~~~~~~~~~~~~~~~~~~~~~~~~

from Zypherus.memory import RawCapture

# Create capture
capture = RawCapture(
    source_url="https://docs.example.com/api",
    content_hash=RawCapture.compute_hash(raw_html),
    raw_html="<html>...</html>",
    cleaned_text="API documentation...",
    metadata={
        "author": "docs team",
        "timestamp": "2026-02-05",
        "version": "1.0"
    }
)

# Store
is_new = orch.raw.store_capture(capture)
print(f"New content: {is_new}")


TASK 3: Extract and Store Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from Zypherus.memory import StructuredPage, Section

# Create sections
sections = [
    Section(
        section_id="intro",
        title="Introduction",
        level=1,
        content="Overview text here...",
        subsections=[]
    ),
    Section(
        section_id="api",
        title="API Reference",
        level=2,
        content="API details...",
        subsections=[]
    ),
]

# Create structured page
page = StructuredPage(
    source_url="https://docs.example.com/api",
    raw_content_hash=capture.content_hash,
    sections=sections,
    version_marker="v1.0"
)

# Store
page_id = orch.structural.store_page(page)
print(f"Page stored as: {page_id}")


TASK 4: Score Content Importance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from Zypherus.memory import ImportanceScorer

scorer = ImportanceScorer()

# Score by content type
type_score = scorer.score_content_type("definition")  # 0.95
type_score = scorer.score_content_type("example")     # 0.45
type_score = scorer.score_content_type("filler")      # 0.1

# Information density
text = "def function(x): return x * 2"
density = scorer.compute_information_density(text)
print(f"Density: {density:.2f}")

# Novelty vs existing
existing = ["def function(x): return x"]
novelty = scorer.compute_novelty(text, existing)
print(f"Novelty: {novelty:.2f}")


TASK 5: Store Semantic Chunks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from Zypherus.memory import ChunkedContent, ImportanceScorer

# Score first
scorer = ImportanceScorer()
importance = scorer.score_content_type("definition")
density = scorer.compute_information_density("definition text")

# Create chunk
chunk = ChunkedContent(
    chunk_id="chunk_abc123",
    source_url="https://docs.example.com",
    page_id="page_xyz",
    content_type="definition",
    text="A comprehensive definition...",
    embedding=[0.1, 0.2, -0.3, ...],  # Optional: add embedding if available
    topic_tags=["api", "reference"],
    importance_score=0.85,
    information_density=density,
)

# Store (only if importance >= threshold)
stored = orch.semantic.store_chunk(chunk)
print(f"Chunk stored: {stored}")

# View stats
stats = orch.semantic.get_stats()
print(f"Total chunks: {stats['total_chunks']}")
print(f"Avg importance: {stats['avg_importance_score']:.2f}")


TASK 6: Evict Low-Value Chunks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Get candidates for eviction
candidates = orch.semantic.get_low_value_chunks()
print(f"Low-value chunks: {len(candidates)}")

# Evict (e.g., if storage budget exceeded)
evicted = orch.semantic.evict_chunks(num_to_evict=10)
print(f"Evicted {evicted} chunks")


TASK 7: Synthesize Learned Facts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from Zypherus.memory import LearnedFact
import time

# Get high-value chunk
chunk = orch.semantic.chunks["chunk_abc123"]

# Create learned fact from chunk
fact = LearnedFact(
    fact_id="fact_xyz",
    statement=chunk.text,
    sources=["chunk_abc123"],
    confidence=0.85,
    expires_at=time.time() + 30*86400,  # 30 days from now
    tags=chunk.topic_tags,
    metadata={"source_url": chunk.source_url}
)

# Store
stored_id = orch.learned.store_fact(fact)
print(f"Fact stored: {stored_id}")


TASK 8: Query Learned Facts
~~~~~~~~~~~~~~~~~~~~~~~~~~

# Get all active (non-expired) facts
active_facts = orch.learned.get_active_facts()
print(f"Active facts: {len(active_facts)}")

# Get facts by tag
kernel_facts = orch.learned.get_facts_by_tag("kernel")
print(f"Kernel-related facts: {len(kernel_facts)}")

# Get expired facts for reconsideration
expired = orch.learned.get_expired_facts()
print(f"Expired facts: {len(expired)}")

# Downgrade old facts (apply decay)
downgraded = orch.learned.downgrade_old_facts(age_days=30)
print(f"Downgraded {downgraded} facts")


TASK 9: Full Ingestion Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from Zypherus.memory import ProgressiveIngestionPipeline

pipeline = ProgressiveIngestionPipeline(orch)

# Ingest entire page through all stages
result = pipeline.ingest_page_full_pipeline(
    source_url="https://example.com/guide",
    raw_html="<html>...</html>",
    cleaned_text="Clean text...",
    sections=[...],  # List of Section objects
    metadata={"author": "...", "date": "..."}
)

print(f"Pipeline result:")
print(f"  Page ID: {result['page_id']}")
print(f"  Chunks stored: {result['chunks_stored']}")
print(f"  Facts synthesized: {result['facts_synthesized']}")
print(f"  Stages: {result['stages_completed']}")


TASK 10: Migrate Existing Memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from Zypherus.memory import ExistingMemoryAdapter
from pathlib import Path

adapter = ExistingMemoryAdapter(
    orch,
    memory_paths={
        "claims": Path("claims.json"),
        "definitions": Path("definitions.json"),
        "memory": Path("memory.json"),
    }
)

# Run migration
results = adapter.run_full_migration()

print(f"Migration complete:")
for layer_result in results["migrations"]:
    print(f"  {layer_result['layer']}: {layer_result['migrated']} migrated")


TASK 11: Export for Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from ace.memory import MemoryLayerBridge
from pathlib import Path

bridge = MemoryLayerBridge(orch)

# Export learned facts back to claims.json format
bridge.export_learned_facts_to_claims(Path("claims_v2.json"))

# Export definitions back to definitions.json format
bridge.export_semantic_chunks_to_definitions(Path("definitions_v2.json"))


TASK 12: Search Across All Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from ace.memory import MemoryLayerBridge

bridge = MemoryLayerBridge(orch)

# Search for content
results = bridge.search_across_layers("overcommit_memory")

print(f"Found {results['count']} results:")
for chunk in results['semantic_chunks']:
    print(f"  [Chunk] {chunk['content_type']}: {chunk['text'][:50]}...")
for fact in results['learned_facts']:
    print(f"  [Fact] confidence={fact['confidence']}: {fact['statement'][:50]}...")


TASK 13: Get Storage Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Full stats from all layers
stats = orch.get_full_stats()

print(f"Raw Captures: {stats['raw_capture']['total_captures']} captures")
print(f"Structural: {stats['structural']['total_pages']} pages")
print(f"Semantic: {stats['semantic']['total_chunks']} chunks")
print(f"Learned: {stats['learned_knowledge']['total_facts']} facts")
print(f"Total Storage: {stats['summary']['total_storage_mb']:.2f} MB")

# Pretty print
orch.print_stats()


TASK 14: Retrieve Content
~~~~~~~~~~~~~~~~~~~~~~~~

# Get captured content by hash
capture = orch.raw.get_capture(content_hash)
if capture:
    print(f"Source: {capture.source_url}")
    print(f"Raw HTML length: {len(capture.raw_html)}")

# Get structured page
page = orch.structural.get_page(page_id)
if page:
    print(f"Sections: {len(page.sections)}")
    print(f"Tables: {len(page.tables)}")

# Get semantic chunk
chunk = orch.semantic.get_chunk(chunk_id)
if chunk:
    print(f"Content type: {chunk.content_type}")
    print(f"Importance: {chunk.importance_score:.2f}")

# Get learned fact
fact = orch.learned.get_fact(fact_id)
if fact:
    print(f"Statement: {fact.statement}")
    print(f"Confidence: {fact.confidence:.2f}")


TASK 15: Advanced Workflow - Site Hierarchy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Process a site's content with proper layer separation

from Zypherus.memory import (
    StorageLayerOrchestrator,
    ProgressiveIngestionPipeline,
)

orch = StorageLayerOrchestrator(Path("./storage"))
pipeline = ProgressiveIngestionPipeline(orch)

pages = [
    {
        "url": "https://docs.example.com/guide",
        "html": "...",
        "text": "...",
        "sections": [...]
    },
    # ... more pages
]

total_captured = 0
total_chunks = 0
total_facts = 0

for page_info in pages:
    result = pipeline.ingest_page_full_pipeline(
        source_url=page_info["url"],
        raw_html=page_info["html"],
        cleaned_text=page_info["text"],
        sections=page_info["sections"],
    )
    
    total_captured += 1
    total_chunks += result["chunks_stored"]
    total_facts += result["facts_synthesized"]

print(f"\\nIngestion Summary:")
print(f"  Pages captured: {total_captured}")
print(f"  Semantic chunks: {total_chunks}")
print(f"  Learned facts: {total_facts}")

# Show final stats
orch.print_stats()


COMMON PATTERNS
===============

Getting important content:
  
  important_chunks = [
      c for c in orch.semantic.chunks.values()
      if c.importance_score >= 0.8
  ]

Exporting for backup:
  
  import json
  
  all_facts = [f.to_dict() for f in orch.learned.get_active_facts()]
  with open("backup_facts.json", "w") as f:
      json.dump(all_facts, f, indent=2)

Finding duplicate content:
  
  seen_hashes = set()
  duplicates = []
  
  for capture in orch.raw.index.values():
      hash_val = capture["content_hash"]
      if hash_val in seen_hashes:
          duplicates.append(capture)
      else:
          seen_hashes.add(hash_val)

Checking storage pressure:
  
  stats = orch.get_full_stats()
  semantic_pct = (
      stats["semantic"]["total_size_mb"] /
      stats["summary"]["total_storage_mb"]
  ) * 100
  
  if semantic_pct > 80:
      print("Semantic layer exceeding 80% - evicting low-value chunks")
      orch.semantic.evict_chunks(num_to_evict=20)
"""

pass
