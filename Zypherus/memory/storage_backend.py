"""
4-Layer Storage Architecture for ACE Memory System.

This implements the canonical separation:
1. Raw Capture Layer (unlimited, write-only, cold)
2. Structural Layer (bounded, reusable)
3. Semantic Layer (controlled, expensive)
4. Learned Knowledge Layer (tiny, high-value)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from logging import getLogger

logger = getLogger(__name__)


# ============================================================================
# LAYER 1: RAW CAPTURE LAYER (Unlimited, Write-Only, Cold)
# ============================================================================

@dataclass
class RawCapture:
    """A single raw capture: HTML, text, metadata with content hash."""
    
    source_url: str
    content_hash: str  # SHA256 of full content
    raw_html: str
    cleaned_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    ingestion_version: str = ""  # e.g., "2026-02-05"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> RawCapture:
        return RawCapture(**d)
    
    @staticmethod
    def compute_hash(content: str) -> str:
        """SHA256 hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()


class RawCaptureStore:
    """Write-only raw capture storage with deduplication by hash."""
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path / "raw_captures"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_path / "index.json"
        self.index: Dict[str, Dict[str, Any]] = self._load_index()
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load or initialize index."""
        if self.index_file.exists():
            with open(self.index_file) as f:
                return json.load(f)
        return {}
    
    def _save_index(self) -> None:
        """Save index."""
        with open(self.index_file, "w") as f:
            json.dump(self.index, f, indent=2)
    
    def store_capture(self, capture: RawCapture) -> bool:
        """Store raw capture, deduplicate by hash. Returns True if new, False if duplicate."""
        content_hash = capture.content_hash
        
        if content_hash in self.index:
            logger.debug(f"Duplicate content skipped (hash: {content_hash[:8]}...)")
            return False
        
        # Write capture to file
        capture_file = self.storage_path / f"{content_hash}.json"
        with open(capture_file, "w") as f:
            json.dump(capture.to_dict(), f, indent=2)
        
        # Update index
        self.index[content_hash] = {
            "source_url": capture.source_url,
            "timestamp": capture.timestamp,
            "version": capture.ingestion_version,
            "file": str(capture_file.relative_to(self.storage_path.parent)),
        }
        self._save_index()
        
        logger.info(f"Raw capture stored: {content_hash[:8]}... from {capture.source_url}")
        return True
    
    def get_capture(self, content_hash: str) -> Optional[RawCapture]:
        """Retrieve raw capture by hash."""
        if content_hash not in self.index:
            return None
        
        capture_file = self.storage_path / f"{content_hash}.json"
        if capture_file.exists():
            with open(capture_file) as f:
                return RawCapture.from_dict(json.load(f))
        return None
    
    def list_captures_by_url(self, url: str) -> List[RawCapture]:
        """Get all captures from a URL."""
        captures = []
        for content_hash, meta in self.index.items():
            if meta["source_url"] == url:
                capture = self.get_capture(content_hash)
                if capture:
                    captures.append(capture)
        return sorted(captures, key=lambda c: c.timestamp)
    
    def get_stats(self) -> Dict[str, Any]:
        """Storage statistics."""
        total_files = len(list(self.storage_path.glob("*.json"))) - 1  # exclude index
        total_size = sum(f.stat().st_size for f in self.storage_path.glob("*.json")) / 1024 / 1024
        return {
            "type": "raw_capture_layer",
            "total_captures": len(self.index),
            "total_files": total_files,
            "total_size_mb": total_size,
            "unique_urls": len(set(m["source_url"] for m in self.index.values())),
        }


# ============================================================================
# LAYER 2: STRUCTURAL LAYER (Bounded, Reusable)
# ============================================================================

@dataclass
class Section:
    """Structured section from a document."""
    section_id: str
    title: str
    level: int  # Heading level (1-6)
    content: str
    subsections: List[str] = field(default_factory=list)  # IDs of child sections


@dataclass
class StructuredPage:
    """Structured extraction from a page: sections, tables, code blocks, links."""
    
    source_url: str
    raw_content_hash: str  # Reference to raw capture
    sections: List[Section] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    code_blocks: List[Dict[str, str]] = field(default_factory=list)
    internal_links: List[Dict[str, str]] = field(default_factory=list)
    version_marker: str = ""  # e.g., "v1.0", "latest"
    extracted_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_url": self.source_url,
            "raw_content_hash": self.raw_content_hash,
            "sections": [asdict(s) for s in self.sections],
            "tables": self.tables,
            "code_blocks": self.code_blocks,
            "internal_links": self.internal_links,
            "version_marker": self.version_marker,
            "extracted_at": self.extracted_at,
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> StructuredPage:
        page = StructuredPage(
            source_url=d["source_url"],
            raw_content_hash=d["raw_content_hash"],
            tables=d.get("tables", []),
            code_blocks=d.get("code_blocks", []),
            internal_links=d.get("internal_links", []),
            version_marker=d.get("version_marker", ""),
            extracted_at=d.get("extracted_at", time.time()),
        )
        page.sections = [
            Section(
                section_id=s["section_id"],
                title=s["title"],
                level=s["level"],
                content=s["content"],
                subsections=s.get("subsections", []),
            )
            for s in d.get("sections", [])
        ]
        return page


class StructuralLayer:
    """Structured knowledge extraction and storage."""
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path / "structural"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_path / "index.json"
        self.index: Dict[str, Dict[str, Any]] = self._load_index()
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        if self.index_file.exists():
            with open(self.index_file) as f:
                return json.load(f)
        return {}
    
    def _save_index(self) -> None:
        with open(self.index_file, "w") as f:
            json.dump(self.index, f, indent=2)
    
    def store_page(self, page: StructuredPage) -> str:
        """Store structured page. Returns page ID."""
        page_id = hashlib.md5(f"{page.source_url}:{page.extracted_at}".encode()).hexdigest()
        
        page_file = self.storage_path / f"{page_id}.json"
        with open(page_file, "w") as f:
            json.dump(page.to_dict(), f, indent=2)
        
        self.index[page_id] = {
            "url": page.source_url,
            "raw_hash": page.raw_content_hash,
            "version": page.version_marker,
            "extracted_at": page.extracted_at,
            "num_sections": len(page.sections),
        }
        self._save_index()
        
        logger.info(f"Structured page stored: {page_id[:8]}... from {page.source_url}")
        return page_id
    
    def get_page(self, page_id: str) -> Optional[StructuredPage]:
        """Retrieve structured page."""
        if page_id not in self.index:
            return None
        
        page_file = self.storage_path / f"{page_id}.json"
        if page_file.exists():
            with open(page_file) as f:
                return StructuredPage.from_dict(json.load(f))
        return None
    
    def get_pages_by_url(self, url: str) -> List[StructuredPage]:
        """Get all structured versions of a URL."""
        pages = []
        for page_id, meta in self.index.items():
            if meta["url"] == url:
                page = self.get_page(page_id)
                if page:
                    pages.append(page)
        return sorted(pages, key=lambda p: p.extracted_at)
    
    def get_stats(self) -> Dict[str, Any]:
        """Storage statistics."""
        total_sections = sum(m.get("num_sections", 0) for m in self.index.values())
        total_size = sum(f.stat().st_size for f in self.storage_path.glob("*.json")) / 1024 / 1024
        return {
            "type": "structural_layer",
            "total_pages": len(self.index),
            "total_sections": total_sections,
            "total_size_mb": total_size,
        }


# ============================================================================
# LAYER 3: SEMANTIC LAYER (Controlled, Expensive)
# ============================================================================

@dataclass
class ChunkedContent:
    """Content chunk for embedding."""
    
    chunk_id: str
    source_url: str
    page_id: str  # Reference to structural page
    content_type: str  # "definition", "explanation", "algorithm", "specification", etc.
    text: str
    embedding: Optional[List[float]] = None
    topic_tags: List[str] = field(default_factory=list)
    importance_score: float = 0.0  # 0.0 - 1.0
    information_density: float = 0.0  # 0.0 - 1.0
    cross_reference_count: int = 0
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> ChunkedContent:
        return ChunkedContent(**d)


class SemanticLayer:
    """Controlled semantic storage with importance scoring."""
    
    def __init__(self, storage_path: Path, importance_threshold: float = 0.5):
        self.storage_path = storage_path / "semantic"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.embeddings_file = self.storage_path / "embeddings.json"
        self.importance_threshold = importance_threshold
        self.chunks: Dict[str, ChunkedContent] = self._load_chunks()
    
    def _load_chunks(self) -> Dict[str, ChunkedContent]:
        if self.embeddings_file.exists():
            with open(self.embeddings_file) as f:
                data = json.load(f)
                return {
                    cid: ChunkedContent.from_dict(chunk)
                    for cid, chunk in data.items()
                }
        return {}
    
    def _save_chunks(self) -> None:
        with open(self.embeddings_file, "w") as f:
            json.dump(
                {cid: chunk.to_dict() for cid, chunk in self.chunks.items()},
                f,
                indent=2,
            )
    
    def score_importance(
        self,
        content_type: str,
        information_density: float,
        cross_reference_count: int,
        is_novel: bool = True,
    ) -> float:
        """Score content importance (0.0 - 1.0)."""
        score = 0.0
        
        # Content type scoring
        high_value_types = {
            "definition": 0.9,
            "specification": 0.85,
            "algorithm": 0.85,
            "rule": 0.8,
            "concept": 0.75,
            "explanation": 0.6,
            "example": 0.4,
            "filler": 0.1,
        }
        score += high_value_types.get(content_type, 0.3)
        
        # Information density (how dense the information is)
        score += information_density * 0.2
        
        # Cross-references (how many other chunks link to it)
        ref_score = min(cross_reference_count / 5.0, 1.0) * 0.2
        score += ref_score
        
        # Novelty bonus
        if is_novel:
            score += 0.1
        
        return min(score, 1.0)
    
    def store_chunk(self, chunk: ChunkedContent) -> bool:
        """Store chunk if importance >= threshold. Returns True if stored."""
        if chunk.importance_score < self.importance_threshold:
            logger.debug(
                f"Chunk {chunk.chunk_id} rejected (importance={chunk.importance_score:.2f})"
            )
            return False
        
        self.chunks[chunk.chunk_id] = chunk
        self._save_chunks()
        
        logger.info(
            f"Semantic chunk stored: {chunk.chunk_id[:8]}... "
            f"(importance={chunk.importance_score:.2f})"
        )
        return True
    
    def get_chunk(self, chunk_id: str) -> Optional[ChunkedContent]:
        """Retrieve chunk."""
        return self.chunks.get(chunk_id)
    
    def search_by_tag(self, tag: str) -> List[ChunkedContent]:
        """Search chunks by topic tag."""
        return [c for c in self.chunks.values() if tag in c.topic_tags]
    
    def get_low_value_chunks(self) -> List[str]:
        """Identify chunks for eviction."""
        candidates = [
            cid for cid, chunk in self.chunks.items()
            if chunk.importance_score < 0.3
        ]
        return sorted(
            candidates,
            key=lambda cid: self.chunks[cid].importance_score,
        )
    
    def evict_chunks(self, num_to_evict: int) -> int:
        """Evict lowest-value chunks. Returns count evicted."""
        candidates = self.get_low_value_chunks()
        evicted = 0
        
        for chunk_id in candidates[:num_to_evict]:
            del self.chunks[chunk_id]
            evicted += 1
        
        self._save_chunks()
        logger.info(f"Evicted {evicted} low-value semantic chunks")
        return evicted
    
    def get_stats(self) -> Dict[str, Any]:
        """Storage statistics."""
        total_size = self.embeddings_file.stat().st_size / 1024 / 1024 if self.embeddings_file.exists() else 0
        avg_importance = (
            sum(c.importance_score for c in self.chunks.values()) / len(self.chunks)
            if self.chunks else 0
        )
        
        return {
            "type": "semantic_layer",
            "total_chunks": len(self.chunks),
            "total_size_mb": total_size,
            "avg_importance_score": avg_importance,
            "importance_threshold": self.importance_threshold,
        }


# ============================================================================
# LAYER 4: LEARNED KNOWLEDGE LAYER (Tiny, High-Value)
# ============================================================================

@dataclass
class LearnedFact:
    """Synthesized, learned fact with provenance."""
    
    fact_id: str
    statement: str
    sources: List[str]  # References to chunks or claims
    confidence: float  # 0.0 - 1.0
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None  # When to reconsider
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> LearnedFact:
        return LearnedFact(**d)
    
    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Check if fact has expired."""
        if self.expires_at is None:
            return False
        return (current_time or time.time()) > self.expires_at


class LearnedKnowledgeLayer:
    """Tiny, high-value learned facts with provenance."""
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path / "learned"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.facts_file = self.storage_path / "facts.json"
        self.facts: Dict[str, LearnedFact] = self._load_facts()
    
    def _load_facts(self) -> Dict[str, LearnedFact]:
        if self.facts_file.exists():
            with open(self.facts_file) as f:
                data = json.load(f)
                return {fid: LearnedFact.from_dict(f) for fid, f in data.items()}
        return {}
    
    def _save_facts(self) -> None:
        with open(self.facts_file, "w") as f:
            json.dump(
                {fid: fact.to_dict() for fid, fact in self.facts.items()},
                f,
                indent=2,
            )
    
    def store_fact(self, fact: LearnedFact) -> str:
        """Store learned fact. Returns fact ID."""
        self.facts[fact.fact_id] = fact
        self._save_facts()
        
        logger.info(f"Learned fact stored: {fact.fact_id} (confidence={fact.confidence:.2f})")
        return fact.fact_id
    
    def get_fact(self, fact_id: str) -> Optional[LearnedFact]:
        """Retrieve fact."""
        return self.facts.get(fact_id)
    
    def get_active_facts(self) -> List[LearnedFact]:
        """Get non-expired facts."""
        now = time.time()
        return [f for f in self.facts.values() if not f.is_expired(now)]
    
    def get_facts_by_tag(self, tag: str) -> List[LearnedFact]:
        """Search facts by tag."""
        return [f for f in self.facts.values() if tag in f.tags]
    
    def get_expired_facts(self) -> List[LearnedFact]:
        """Identify facts to reconsidera."""
        now = time.time()
        return [f for f in self.facts.values() if f.is_expired(now)]
    
    def downgrade_old_facts(self, age_days: int = 30) -> int:
        """Mark old facts for reconsideration by lowering confidence."""
        now = time.time()
        age_seconds = age_days * 86400
        downgraded = 0
        
        for fact in self.facts.values():
            if (now - fact.created_at) > age_seconds:
                original = fact.confidence
                fact.confidence *= 0.9  # Decay by 10%
                if original > 0.5 and fact.confidence <= 0.5:
                    logger.info(f"Downgraded fact: {fact.fact_id}")
                    downgraded += 1
        
        if downgraded > 0:
            self._save_facts()
        
        return downgraded
    
    def get_stats(self) -> Dict[str, Any]:
        """Storage statistics."""
        total_size = self.facts_file.stat().st_size / 1024 if self.facts_file.exists() else 0
        active = self.get_active_facts()
        expired = self.get_expired_facts()
        avg_confidence = sum(f.confidence for f in active) / len(active) if active else 0
        
        return {
            "type": "learned_knowledge_layer",
            "total_facts": len(self.facts),
            "active_facts": len(active),
            "expired_facts": len(expired),
            "avg_confidence": avg_confidence,
            "size_kb": total_size,
        }


# ============================================================================
# ORCHESTRATOR: All 4 Layers
# ============================================================================

class StorageLayerOrchestrator:
    """Manage all 4 layers together."""
    
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.raw = RawCaptureStore(self.storage_path)
        self.structural = StructuralLayer(self.storage_path)
        self.semantic = SemanticLayer(self.storage_path)
        self.learned = LearnedKnowledgeLayer(self.storage_path)
    
    def get_full_stats(self) -> Dict[str, Any]:
        """Get stats from all 4 layers."""
        return {
            "raw_capture": self.raw.get_stats(),
            "structural": self.structural.get_stats(),
            "semantic": self.semantic.get_stats(),
            "learned_knowledge": self.learned.get_stats(),
            "summary": {
                "total_storage_mb": (
                    self.raw.get_stats()["total_size_mb"]
                    + self.structural.get_stats()["total_size_mb"]
                    + self.semantic.get_stats()["total_size_mb"]
                    + self.learned.get_stats()["size_kb"] / 1024
                ),
            },
        }
    
    def print_stats(self) -> None:
        """Print formatted storage statistics."""
        stats = self.get_full_stats()
        
        print("\n" + "=" * 70)
        print("4-LAYER STORAGE ARCHITECTURE STATISTICS")
        print("=" * 70)
        
        print(f"\n📦 Layer 1: Raw Capture (Unlimited, Cold)")
        print(f"   Captures: {stats['raw_capture']['total_captures']}")
        print(f"   Unique URLs: {stats['raw_capture']['unique_urls']}")
        print(f"   Size: {stats['raw_capture']['total_size_mb']:.2f} MB")
        
        print(f"\n📋 Layer 2: Structural (Bounded, Reusable)")
        print(f"   Pages: {stats['structural']['total_pages']}")
        print(f"   Sections: {stats['structural']['total_sections']}")
        print(f"   Size: {stats['structural']['total_size_mb']:.2f} MB")
        
        print(f"\n🧠 Layer 3: Semantic (Controlled, Expensive)")
        print(f"   Chunks: {stats['semantic']['total_chunks']}")
        print(f"   Avg Importance: {stats['semantic']['avg_importance_score']:.2f}")
        print(f"   Threshold: {stats['semantic']['importance_threshold']}")
        print(f"   Size: {stats['semantic']['total_size_mb']:.2f} MB")
        
        print(f"\n💎 Layer 4: Learned Knowledge (Tiny, High-Value)")
        print(f"   Facts: {stats['learned_knowledge']['total_facts']}")
        print(f"   Active: {stats['learned_knowledge']['active_facts']}")
        print(f"   Expired: {stats['learned_knowledge']['expired_facts']}")
        print(f"   Avg Confidence: {stats['learned_knowledge']['avg_confidence']:.2f}")
        print(f"   Size: {stats['learned_knowledge']['size_kb']:.2f} KB")
        
        print(f"\n📊 TOTAL STORAGE: {stats['summary']['total_storage_mb']:.2f} MB")
        print("=" * 70 + "\n")
